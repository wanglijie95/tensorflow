#include "tensorflow/core/kernels/replication_counter.h"
#include <iostream>
namespace tensorflow{

void ReplicationCounter::ResetUnlock(){
  pull_counter = 0;
  send_counter = 0;
  pull_worker_set.clear();
}


ReplicationCounter* ReplicationCounterManager::GetOrCreateCounter(string var_name){
  mutex_lock l(mu_);
  auto iter = counters_.find(var_name);
  if (iter == counters_.end()){
    iter = counters_.emplace(std::piecewise_construct,
                             std::forward_as_tuple(var_name),
                             std::forward_as_tuple(var_name)).first;
  }
  return &(iter->second);
}


int ReplicationCounterManager::number_counters(){
  mutex_lock l(mu_);
  return counters_.size();
}

// The global replication counter manager.
ReplicationCounterManager g_replication_counter_manager;

} // end namespace tensorflow



