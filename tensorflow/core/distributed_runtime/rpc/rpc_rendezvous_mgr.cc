/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "grpc++/support/byte_buffer.h"

#include <unordered_set>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class RpcRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  RpcRemoteRendezvous(const WorkerEnv* env, int64 step_id)
      : BaseRemoteRendezvous(env, step_id) {}

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;
  
  void SendToRemoteAsync(const Rendezvous::ParsedKey& parsed,
                        const Rendezvous::Args& send_args,
                        const int64 global_step,
                        const string replication_name,
                        const Tensor& val,
                        StatusCallback done) override;

 private:
  ~RpcRemoteRendezvous() override {}

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRemoteRendezvous);
};

// Used only to retrieve tensors from remote processes.
class RpcRecvTensorCall : public BaseRecvTensorCall {
 public:
  RpcRecvTensorCall() : wi_(nullptr), dst_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* dst_device,
            const Rendezvous::Args& recv_args, Rendezvous::DoneCallback done) {
    wi_ = wi;
    alloc_attrs_ = alloc_attrs;
    dst_device_ = dst_device;
    recv_args_ = recv_args;
    done_ = std::move(done);
    req_.set_step_id(step_id);
    req_.set_rendezvous_key(key.data(), key.size());
    req_.set_request_id(GetUniqueRequestId());
  }

  void Reset(WorkerCacheInterface* wc) {
    wc->ReleaseWorker(src_worker_, wi_);
    wi_ = nullptr;
    alloc_attrs_ = AllocatorAttributes();
    dst_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~RpcRecvTensorCall() override {
    // Since only the RpcRecvTensorFreeList will delete an
    // RpcRecvTensorCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcRecvTensorCall destructor.";
  }

  void Start(std::function<void()> recv_done) override {
    StartRTCall(std::move(recv_done));
  }

  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }

  const Tensor& tensor() const { return resp_.tensor(); }

  bool is_dead() const { return resp_.metadata().is_dead(); }

  Device* dst_device() const { return dst_device_; }
  const Rendezvous::Args& recv_args() const { return recv_args_; }
  const Rendezvous::DoneCallback& done() const { return done_; }

 private:
  friend class RpcRemoteRendezvous;

  // Start the main RecvTensor call, checking for an async abort.
  void StartRTCall(std::function<void()> recv_done) {
    resp_.InitAlloc(dst_device_, alloc_attrs_);
    using namespace std::placeholders;
    StatusCallback cb = std::bind(
        [this](std::function<void()> recv_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          recv_done();
        },
        std::move(recv_done), _1);
    wi_->RecvTensorAsync(&opts_, &req_, &resp_, std::move(cb));
  }

  string src_worker_;
  string src_rel_device_;
  WorkerInterface* wi_;
  AllocatorAttributes alloc_attrs_;
  Device* dst_device_;
  CallOptions opts_;
  RecvTensorRequest req_;
  TensorResponse resp_;
  Rendezvous::Args recv_args_;
  Rendezvous::DoneCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RpcRecvTensorCall);
};

class RpcRecvTensorFreeList {
 public:
  RpcRecvTensorFreeList() {}
  ~RpcRecvTensorFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  RpcRecvTensorCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        RpcRecvTensorCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new RpcRecvTensorCall;
  }

  void Release(RpcRecvTensorCall* obj, WorkerCacheInterface* wc) {
    obj->Reset(wc);
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static const int kMaxObjects = 1000;

  mutex mu_;
  std::vector<RpcRecvTensorCall*> objects_ GUARDED_BY(mu_);
};

static RpcRecvTensorFreeList* get_call_freelist() {
  static RpcRecvTensorFreeList* call_freelist = new RpcRecvTensorFreeList();
  return call_freelist;
}

void RpcRemoteRendezvous::RecvFromRemoteAsync(
    const Rendezvous::ParsedKey& parsed, const Rendezvous::Args& recv_args,
    DoneCallback done) {
  CHECK(is_initialized());
  Status s;

  // Prepare a RecvTensor call that can handle being aborted.
  RpcRecvTensorCall* call = get_call_freelist()->New();

  // key.src_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.src_device, &call->src_worker_,
                                        &call->src_rel_device_)) {
    s = errors::Internal(parsed.src_device,
                         " is invalid remote source device.");
  }
  WorkerSession* sess = session();
  WorkerInterface* rwi = sess->worker_cache->CreateWorker(call->src_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->src_worker_);
  }

  Device* dst_device;
  if (s.ok()) {
    s = sess->device_mgr()->LookupDevice(parsed.dst_device, &dst_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache->ReleaseWorker(call->src_worker_, rwi);
    }
    get_call_freelist()->Release(call, sess->worker_cache.get());
    done(s, Args(), recv_args, Tensor{}, false);
    return;
  }

  call->Init(rwi, step_id_, parsed.FullKey(), recv_args.alloc_attrs, dst_device,
             recv_args, std::move(done));

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call);

  // Start "call".
  Ref();
  call->Start([this, call]() {
    // Removes "call" from active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    call->done()(s, Args(), call->recv_args(), call->tensor(), call->is_dead());
    session()->worker_cache->ReleaseWorker(call->src_worker_, call->wi_);
    call->wi_ = nullptr;
    get_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
  });
}

// Used only to send replications to remote processes.
class RpcSendReplicationCall : public BaseSendReplicationCall {
 public:
  RpcSendReplicationCall() : wi_(nullptr), src_device_(nullptr) {}

  void Init(WorkerInterface* wi, int64 step_id, StringPiece key,
            AllocatorAttributes alloc_attrs, Device* src_device,
            const Rendezvous::Args& send_args, const int64 global_step,
            const string replication_name, const Tensor& val,
            Rendezvous::StatusCallback done) {
    wi_ = wi;
    alloc_attrs_ = alloc_attrs;
    src_device_ = src_device;
    send_args_ = send_args;
    done_ = std::move(done);
    const bool on_host = alloc_attrs_.on_host();
    {
      // Non_DMA cases.
      if(src_device_->tensorflow_gpu_device_info() && (!on_host)){
#if GOOGLE_CUDA
        const DeviceContext* send_dev_context = send_args_.device_context;
        AllocatorAttributes gpu_alloc_attrs;
        gpu_alloc_attrs.set_gpu_compatible(true);
        gpu_alloc_attrs.set_on_host(true);
        Allocator* alloc = src_device_->GetAllocator(gpu_alloc_attrs);
        Tensor* copy = new Tensor(alloc, val.dtype(), val.shape());
        CHECK(send_dev_context)
            << "send dev name: " << src_device_->name()
            << " gpu_info: " << src_device_->tensorflow_gpu_device_info();
        // "val" is on a GPU. Uses GPUUtil to fill the copy on host.
        ::grpc::ByteBuffer* request = &req_;
        StatusCallback copy_ready = [request, copy, key, global_step,
                                    replication_name, &status_](const Status& s) {
          // The value is now ready to be returned on the wire.
          tensorflow::grpc::EncodeTensorToByteBuffer(key, global_step, replication_name, *copy, request);
          status_.Update(s);
          delete copy;
        };

        tensorflow::GPUUtil::CopyGPUTensorToCPU(src_device_, send_dev_context, &val, copy,
                            copy_ready);
#else
        status_.Update(errors::Internal("No GPU device in process"));
#endif // GOOGLE_CUDA
      } else {
        tensorflow::grpc::EncodeTensorToByteBuffer(key, global_step, replication_name, val, &req_);
      }
    }
  }

  void Reset(WorkerCacheInterface* wc) {
    wc->ReleaseWorker(dst_worker_, wi_);
    wi_ = nullptr;
    alloc_attrs_ = AllocatorAttributes();
    src_device_ = nullptr;
    // We don't clear opts_ and assume that Init will set up the state for
    // opts_ appropriately.
    req_.Clear();
    resp_.Clear();
    {
      mutex_lock l(mu_);
      status_ = Status::OK();
    }
    done_ = nullptr;
  }

  ~RpcSendReplicationCall() override {
    // Since only the RpcSendReplicationFreeList will delete an
    // RpcSendReplicationCall, and it always sets this->wi_ to null when
    // a call object is released to it, we can assert that this->wi_ is
    // always null at the point of deletion.
    CHECK_EQ(static_cast<WorkerInterface*>(nullptr), wi_)
        << "Leaking WorkerInterface in RpcSendReplicationCall destructor.";
  }

  void Start(std::function<void()> send_done) override {
    StartSRCall(std::move(send_done));
  }

  void StartAbort(const Status& s) override {
    {
      mutex_lock l(mu_);
      status_.Update(s);
    }
    opts_.StartCancel();
  }

  Status status() const override {
    mutex_lock l(mu_);
    return status_;
  }


  Device* src_device() const { return src_device_; }
  const Rendezvous::Args& send_args() const { return send_args_; }
  const Rendezvous::StatusCallback& done() const { return done_; }

 private:
  friend class RpcRemoteRendezvous;

  // Start the main SendReplication call, checking for an async abort.
  void StartSRCall(std::function<void()> send_done) {
    using namespace std::placeholders;
    StatusCallback cb = std::bind(
        [this](std::function<void()> send_done,
               // Begin unbound arguments.
               const Status& s) {
          if (!s.ok()) {
            mutex_lock l(mu_);
            status_.Update(s);
          }
          send_done();
        },
        std::move(send_done), _1);
    wi_->SendReplicationAsync(&opts_, &req_, &resp_, std::move(cb));
  }

  string dst_worker_;
  string dst_rel_device_;
  WorkerInterface* wi_;
  AllocatorAttributes alloc_attrs_;
  Device* src_device_;
  CallOptions opts_;
  ::grpc::ByteBuffer req_;
  SendReplicationResponse resp_;
  Rendezvous::Args send_args_;
  Rendezvous::StatusCallback done_;

  mutable mutex mu_;
  Status status_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(RpcSendReplicationCall);
};

class RpcSendReplicationFreeList {
 public:
  RpcSendReplicationFreeList() {}
  ~RpcSendReplicationFreeList() {
    for (size_t i = 0; i < objects_.size(); i++) {
      delete objects_[i];
    }
  }

  RpcSendReplicationCall* New() {
    {
      mutex_lock l(mu_);
      if (!objects_.empty()) {
        RpcSendReplicationCall* result = objects_.back();
        objects_.pop_back();
        return result;
      }
    }
    return new RpcSendReplicationCall;
  }

  void Release(RpcSendReplicationCall* obj, WorkerCacheInterface* wc) {
    obj->Reset(wc);
    {
      mutex_lock l(mu_);
      if (objects_.size() < kMaxObjects) {
        objects_.push_back(obj);
        return;
      }
    }
    delete obj;
  }

 private:
  static const int kMaxObjects = 1000;

  mutex mu_;
  std::vector<RpcSendReplicationCall*> objects_ GUARDED_BY(mu_);
};

static RpcSendReplicationFreeList* get_replication_call_freelist() {
  static RpcSendReplicationFreeList* call_freelist = new RpcSendReplicationFreeList();
  return call_freelist;
}


void RpcRemoteRendezvous::SendToRemoteAsync(const Rendezvous::ParsedKey& parsed,
                                            const Rendezvous::Args& send_args,
                                            const int64 global_step,
                                            const string replication_name,
                                            const Tensor& val,
                                            StatusCallback done) {
  CHECK(is_initialized());
  Status s;

  // Prepare a SendReplication call that can handle being aborted.
  RpcSendReplicationCall* call = get_replication_call_freelist()->New();

  // key.dst_device identifies a remote device.
  if (!DeviceNameUtils::SplitDeviceName(parsed.dst_device, &call->dst_worker_,
                                        &call->dst_rel_device_)) {
    s = errors::Internal(parsed.dst_device,
                         " is invalid remote destination device.");
  }
  WorkerSession* sess = session();
  WorkerInterface* rwi = sess->worker_cache->CreateWorker(call->dst_worker_);
  if (s.ok() && rwi == nullptr) {
    s = errors::Internal("No worker known as ", call->dst_worker_);
  }

  Device* src_device;
  if (s.ok()) {
    s = sess->device_mgr->LookupDevice(parsed.src_device, &src_device);
  }
  if (!s.ok()) {
    if (rwi != nullptr) {
      sess->worker_cache->ReleaseWorker(call->dst_worker_, rwi);
    }
    get_replication_call_freelist()->Release(call, sess->worker_cache.get());
    done(s);
    return;
  }

  call->Init(rwi, step_id_, parsed.FullKey(), send_args.alloc_attrs, src_device,
             send_args, global_step, replication_name, val, std::move(done));
  s = call->status();
  if (!s.ok()){
    call->done()(s);
    session()->worker_cache->ReleaseWorker(call->dst_worker_, call->wi_);
    call->wi_ = nullptr;
    get_replication_call_freelist()->Release(call, session()->worker_cache.get());
    return;
  }

  // Record "call" in active_ so that it can be aborted cleanly.
  RegisterCall(call);

  // Start "call".
  Ref();
  call->Start([this, call]() {
    // Removes "call" from send_active_. Prevent StartAbort().
    DeregisterCall(call);
    // If StartAbort was called prior to DeregisterCall, then the
    // current status should be bad.
    Status s = call->status();
    call->done()(s);
    session()->worker_cache->ReleaseWorker(call->dst_worker_, call->wi_);
    call->wi_ = nullptr;
    get_replication_call_freelist()->Release(call, session()->worker_cache.get());
    Unref();
  });
}

}  // namespace

RpcRendezvousMgr::RpcRendezvousMgr(const WorkerEnv* env)
    : BaseRendezvousMgr(env) {}

BaseRemoteRendezvous* RpcRendezvousMgr::Create(int64 step_id,
                                               const WorkerEnv* worker_env) {
  return new RpcRemoteRendezvous(worker_env, step_id);
}

}  // end namespace tensorflow
