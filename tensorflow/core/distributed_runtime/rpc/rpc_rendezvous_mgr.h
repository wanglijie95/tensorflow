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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_RPC_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_RPC_RENDEZVOUS_MGR_H_

#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/platform/macros.h"

namespace grpc {
class ByteBuffer;
}  // namespace grpc

namespace tensorflow {

class DeviceMgr;

// RendezvousMgr keeps track of a set of local rendezvous instances.
// All tensors sent by this worker are buffered in a RendezvousMgr
// until the tensor is received.  Each global unique "step_id"
// corresponds to one local rendezvous instance managed by a
// RendezvousMgr.
//
// E.g.,
//   Rendezvous* rendez = worker_env->rendezvous_mgr->Find(0x8935);
//   fork execution of an graph executor using "rendez"  on thread 1;
//   fork execution of another graph executor using "rendez" on thread 2;
//   ...
//   join threads 1 and 2;
//
// In the example above, execution in thread 1 and 2 communicates with
// each other by send/recv operations through the "rend".
//
// Tensors sent and recved through rendezvous managed by this
// RendezvousMgr must have keys generated by Rendezvous::CreateKey.
class RpcRendezvousMgr : public BaseRendezvousMgr {
 public:
  explicit RpcRendezvousMgr(const WorkerEnv* env);

 protected:
  BaseRemoteRendezvous* Create(int64 step_id, const WorkerEnv* worker_env);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RpcRendezvousMgr);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_RPC_RENDEZVOUS_MGR_H_
