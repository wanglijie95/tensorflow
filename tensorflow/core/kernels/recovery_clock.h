#ifndef TENSORFLOW_CORE_KERNELS_RECOVERY_CLOCK_H_
#define TENSORFLOW_CORE_KERNELS_RECOVERY_CLOCK_H_

#include <list>
#include <map>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"


namespace tensorflow{

class RecoveryClock : public ResourceBase {
public:
  typedef std::function<void(const int32&)> CallbackWithInt;
  typedef std::function<void(const std::vector<string>&)> CallbackWithString;

  RecoveryClock(int32 total_num_replicas);
  void TryGetLastestWorker(int32 replica_index, int64 local_step,
                           OpKernelContext* ctx, CallbackWithInt callback);
  void TryGetRecoveredVars(int32 replica_index, OpKernelContext* ctx,
                           CallbackWithString callback);
  // Clear the RecoveryClock
  // Always be used to init the RecoveryClock before reuse it.
  // void Clear(){
  //   clocks_.clear();
  //   var_worker_.clear();
  //   get_worker_attempts_.clear();
  //   get_vars_attempts_.clear();
  // }
  string DebugString() override { return "A recovery clock"; }

private:
  // The number of workers in cluster
  int32 total_num_replicas_;
  // The clock for each worker.<worker_index, worker_clock>
  std::map<int32, int64> clocks_ GUARDED_BY(mu_);

  struct ShadowInfo{
    string var_name;
    int64 var_step;
    int32 replica_index;
  };
  
  
  // Record each worker's shadows number
  // Used by `GetRecoveredVars`
  std::map<int32, int> shadow_num_ GUARDED_BY(mu_);
  // The map for geting the lastest worker from workers who has the var.
  // <var_name, ShadowInfo>. This map is just used by `GetRecoveredVars`
  std::map<string, ShadowInfo> shadow_info_ GUARDED_BY(mu_);
  // mutex of for this class
  mutex mu_;

protected:

  struct Attempt;
  struct Attempt {
    // Maybe the `replica_index` is useful just for `GetRecoveredVars`
    int32 replica_index;
    // `done_callback` is for `GetLastestWorker` attempts
    CallbackWithInt done_callback;
    // `done_callback` is for `GetRecoveredVars` attempts
    CallbackWithString done_callback2;
    OpKernelContext* context;
    CancellationManager* cancellation_manager;
    CancellationToken cancellation_token;
    bool is_cancelled;

    Attempt(int32 replica_index, 
            CallbackWithInt done_callback, 
            CallbackWithString done_callback2,
            OpKernelContext* context,
            CancellationManager* cancellation_manager,
            CancellationToken cancellation_token)
            : replica_index(replica_index),
              done_callback(done_callback),
              done_callback2(done_callback2),
              context(context),
              cancellation_manager(cancellation_manager),
              cancellation_token(cancellation_token),
              is_cancelled(false) {}
  };

  // The GetLastestWorker attempts
  std::list<Attempt> get_worker_attempts_ GUARDED_BY(mu_);
  // The GetRecoveredVars attempts
  std::list<Attempt> get_vars_attempts_ GUARDED_BY(mu_);
  //Cancel function
  void GetWorkerCancel(CancellationManager* cancellation_manager, CancellationToken token);
  void GetVarsCancel(CancellationManager* cancellation_manager, CancellationToken token);
};

}

#endif
