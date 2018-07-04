#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/kernels/recovery_clock.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"


namespace tensorflow{

class RecoveryClockOp : public ResourceOpKernel<RecoveryClock> {
public:
  explicit RecoveryClockOp(OpKernelConstruction* context) : ResourceOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("total_num_replicas", &total_num_replicas_));
  }

private:
  Status CreateResource(RecoveryClock** ret) override
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    RecoveryClock* clock = new RecoveryClock(total_num_replicas_);
    if (clock == nullptr) {
      return errors::ResourceExhausted("Failed to allocate report clock");
    }
    *ret = clock;
    return Status::OK();
  }
    
protected:
  int32 total_num_replicas_;
};
REGISTER_KERNEL_BUILDER(Name("RecoveryClock").Device(DEVICE_CPU), RecoveryClockOp);



class GetLastestWorkerOp : public AsyncOpKernel {
public:
  explicit GetLastestWorkerOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("replica_index", &replica_index_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final {
    RecoveryClock* clock;
    OP_REQUIRES_OK_ASYNC(ctx, GetResourceFromContext(ctx, "handle", &clock), callback);
    
    // Get input local_step
    const Tensor* local_step_tensor;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("local_step", &local_step_tensor), callback);
    if (!TensorShapeUtils::IsScalar(local_step_tensor->shape())) {
      ctx->CtxFailureWithWarning(errors::InvalidArgument(
          "Argument num_required must be scalar, but had bad shape ",
          local_step_tensor->shape().DebugString()));
    }

    ComputeAsync(clock, local_step_tensor->scalar<int64>()(),
                 ctx, [callback, clock]() {
      clock->Unref();
      callback();
    });
  }

protected:
  int32 replica_index_;

  void ComputeAsync(RecoveryClock* clock, int64 local_step,
                    OpKernelContext* ctx, DoneCallback callback) {
    clock->TryGetLastestWorker(replica_index_, local_step, ctx, 
                              [ctx, callback](const int32& lastest_worker) {
      if (!ctx->status().ok()) {
        callback();
        return;
      }
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                    ctx->allocate_output(0, TensorShape({}), &output));
      auto output_tensor = output->tensor<int32, 0>();
      output_tensor(0) = lastest_worker;
      callback();
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("GetLastestWorker").Device(DEVICE_CPU), GetLastestWorkerOp);

class GetRecoveredVarsOp : public AsyncOpKernel {
public:
  explicit GetRecoveredVarsOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("replica_index", &replica_index_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback callback) final {
    RecoveryClock* clock;
    // Get the handle of recovery_clock
    OP_REQUIRES_OK_ASYNC(ctx, GetResourceFromContext(ctx, "handle", &clock), callback);

    ComputeAsync(clock, ctx, [callback, clock]() {
      clock->Unref();
      callback();
    });
  }

protected:
  int32 replica_index_;

  void ComputeAsync(RecoveryClock* clock, OpKernelContext* ctx, DoneCallback callback) {
    clock->TryGetRecoveredVars(replica_index_, ctx, 
                              [ctx, callback](const std::vector<string>& var_names) {
      if (!ctx->status().ok()) {
        callback();
        return;
      }
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx,
                    ctx->allocate_output(0, TensorShape({var_names.size()}), &output));
      auto output_tensor = output->tensor<string, 1>();
      for(int32 i = 0; i < var_names.size(); ++i){
        output_tensor(i) = var_names[i];
      }
      callback();
    });
  }
};

REGISTER_KERNEL_BUILDER(Name("GetRecoveredVars").Device(DEVICE_CPU), GetRecoveredVarsOp);


// class RecoveryClockClearOp : public OpKernel {
//   public:
//     explicit RecoveryClockClearOp(OpKernelConstruction* context) : OpKernel(context) { }

//     void Compute(OpKernelContext* ctx) override {
//       RecoveryClock* clock;
//       OP_REQUIRES_OK(ctx, GetResourceFromContext(ctx, "handle", &clock));
//       clock->Clear();
//     }
// };
// REGISTER_KERNEL_BUILDER(Name("RecoveryClockClear").Device(DEVICE_CPU), RecoveryClockClearOp);

}
