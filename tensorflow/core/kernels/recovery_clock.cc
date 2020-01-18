#include "tensorflow/core/kernels/recovery_clock.h"

namespace tensorflow {

RecoveryClock::RecoveryClock(int32 total_num_replicas) 
            : total_num_replicas_(total_num_replicas) {}

// Get the index of worker who has the lastest parameter.
// This is a blocking function. 
// The function block until collect `N` local_step, each worker
// will send a local_step to RecoveryClock.
void RecoveryClock::TryGetLastestWorker(int32 replica_index,
                                        int64 local_step,
                                        OpKernelContext* ctx, 
                                        CallbackWithInt callback) { 
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { GetWorkerCancel(cm, token); });
    if (!already_cancelled) {
      // Insert the <replica_index, local_step> into clocks_.
      // Add the attempt into get_worker_attemtps_.
      clocks_[replica_index] = local_step;
      get_worker_attempts_.emplace_back(
                      replica_index, callback, nullptr, ctx, cm, token);

      // Have received `N` local_step.
      // Find the max value of clocks_, and get the index.
      if (clocks_.size() >= total_num_replicas_) {
        int32 max_key = -1;
        int64 max_value = -1;
        std::map<int32,int64>::iterator iter;
        for (iter=clocks_.begin(); iter!=clocks_.end(); iter++) {
          if (iter->second > max_value) {
            max_value = iter->second;
            max_key = iter->first;
          }
        }
        // Flush all attempts, all attempts return the `max_key`
        for(Attempt& attempt : get_worker_attempts_) {
          if (attempt.cancellation_token != CancellationManager::kInvalidToken) {
            attempt.cancellation_manager->DeregisterCallback(attempt.cancellation_token);
          }
          attempt.done_callback(max_key);
        }
        get_worker_attempts_.clear();
        clocks_.clear();
      } 
    }
  }

  if (already_cancelled) {
    ctx->SetStatus(errors::Cancelled("GetLastestWorker operation was cancelled"));
    callback(-1);
  }
}

// Cancel `GetLastestWorker`.
void RecoveryClock::GetWorkerCancel(CancellationManager* cancellation_manager, 
                                            CancellationToken token) {
  CallbackWithInt callback = nullptr;
  {
    mutex_lock lock(mu_);
    for (Attempt& attempt : get_worker_attempts_) {
      if (attempt.cancellation_manager == cancellation_manager 
                    &&  attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          attempt.context->SetStatus(
              errors::Cancelled("GetLastestWorker operation was cancelled"));
          std::swap(callback, attempt.done_callback);
        }
      }
    }
  }
  if (callback) {
    callback(-1);
  }
}

void RecoveryClock::TryGetRecoveredVars(int32 replica_index,
                                     OpKernelContext* ctx, 
                                     CallbackWithString callback) { 
  CancellationManager* cm = ctx->cancellation_manager();
  CancellationToken token = cm->get_cancellation_token();
  bool already_cancelled;
  {
    mutex_lock l(mu_);
    already_cancelled = !cm->RegisterCallback(
        token, [this, cm, token]() { GetVarsCancel(cm, token); });
    if (!already_cancelled) {
      const Tensor& var_names = ctx->input(1);
      const Tensor& var_steps = ctx->input(2);
      const int var_names_num = static_cast<int>(var_names.NumElements());

      // Check num
      if (var_names_num != static_cast<int>(var_steps.NumElements())){
        ctx->SetStatus(errors::Internal("Variable names number and steps number are not equal"));
        callback(std::vector<string>());
      }

      shadow_num_[replica_index] = var_names_num;

      // Get input var_names and var_steps.
      const auto& var_names_flat = var_names.flat<string>();
      const auto& var_steps_flat = var_steps.flat<int64>();
      
      for(int i = 0; i < var_names_num; ++i){
        string name = var_names_flat(i);
        int64 step = var_steps_flat(i);
        // check the `name` whether in map.
        if(shadow_info_.count(name)){
          // `name` is in map, compare the var_step
          ShadowInfo& shadow = shadow_info_[name];
          if(step > shadow.var_step){
            // worker `replica_index` has the lastest var, update
            shadow.var_step = step;
            shadow.replica_index = replica_index;
          }
        } else {
          // `name` is not in map, insert
          ShadowInfo temp;
          temp.var_name = name;
          temp.var_step = step;
          temp.replica_index = replica_index;
          shadow_info_[name] = temp;
        }
      }

      get_vars_attempts_.emplace_back(
                        replica_index, nullptr, callback, ctx, cm, token);
      
      // Have received `N`.
      // Flush all attempts.
      if (shadow_num_.size() >= total_num_replicas_) {
        // For more efficient, we sort out the worker(`replica_index`) and
        // all vars that it's responsible for recovering.
        // <replica_index, (var1, var2, ...)>
        std::map<int32, std::vector<string>> worker_vars;
        for (auto iter=shadow_info_.begin(); iter!=shadow_info_.end(); iter++){
          string var_name = iter->first;
          int32 worker_index = iter->second.replica_index;
          if(worker_vars.count(worker_index)){
            worker_vars[worker_index].emplace_back(var_name);
          } else {
            std::vector<string> tmp_vector;
            worker_vars[worker_index] = tmp_vector;
            worker_vars[worker_index].emplace_back(var_name);
          }
        }

        // Now we flush the get_vars_attempts
        for(Attempt& attempt : get_vars_attempts_) {
          if (attempt.cancellation_token != CancellationManager::kInvalidToken) {
            attempt.cancellation_manager->DeregisterCallback(attempt.cancellation_token);
          }
          attempt.done_callback2(worker_vars[attempt.replica_index]);
        }

        // Clear related map and list.
        get_vars_attempts_.clear();
        shadow_num_.clear();
        shadow_info_.clear();
      }
    }
  }

  if (already_cancelled) {
    ctx->SetStatus(errors::Cancelled("GetLastestWorker operation was cancelled"));
    callback(std::vector<string>());
  }
}

// Cancel `GetRecoveredVars`.
void RecoveryClock::GetVarsCancel(CancellationManager* cancellation_manager, 
                                            CancellationToken token) {
  CallbackWithString callback = nullptr;
  {
    mutex_lock lock(mu_);
    for (Attempt& attempt : get_vars_attempts_) {
      if (attempt.cancellation_manager == cancellation_manager 
                    &&  attempt.cancellation_token == token) {
        if (!attempt.is_cancelled) {
          attempt.is_cancelled = true;
          attempt.context->SetStatus(
              errors::Cancelled("GetLastestWorker operation was cancelled"));
          std::swap(callback, attempt.done_callback2);
        }
      }
    }
  }
  if (callback) {
    callback(std::vector<string>());
  }
}

}
