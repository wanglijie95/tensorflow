#include "tensorflow/core/kernels/replication_counter.h"
#include <iostream>
namespace tensorflow{

void ReplicationCounter::ResetUnlock(){
  pull_counter = 0;
  send_counter = 0;
  pull_worker_set.clear();
}

//Some functions for counters_ (insert counter, delete counter, get counter)
//When insert, If counter name is existed, do nothing.
void ReplicationCounterManager::InsertCounter(ReplicationCounter* counter){
  mutex_lock l(mu_);
  if (GetCounterUnlock(counter->var_name) == nullptr){
    counters_[counter->var_name] = counter;
  }
}

void ReplicationCounterManager::InsertCounter(string var_name){
  mutex_lock l(mu_);
  if (GetCounterUnlock(var_name) == nullptr){
    ReplicationCounter* counter = new ReplicationCounter(var_name);
    counters_[counter->var_name] = counter;
  } 
}

void ReplicationCounterManager::DeleteCounter(string var_name){
  mutex_lock l(mu_);
  ReplicationCounter* old_counter = GetCounterUnlock(var_name);
  if (old_counter != nullptr){
    counters_.erase(var_name);
    delete old_counter;
  }
}

ReplicationCounter* ReplicationCounterManager::GetOrCreateCounter(string var_name){
  mutex_lock l(mu_);
  ReplicationCounter* result = GetCounterUnlock(var_name);
  if (result == nullptr){
    std::cout << "Create a new counter, name is : " << var_name << std::endl;
    ReplicationCounter* counter = new ReplicationCounter(var_name);
    counters_[counter->var_name] = counter;
    return counter;
  }
  return result;
}

// This function is used by `InsertCounter` and `DeleteCounter`.
// When call this function, the lock is acquired, so we don't
// need to acquire the lock agagin. Otherwise it causes a deadlock.
ReplicationCounter* ReplicationCounterManager::GetCounterUnlock(string var_name){
  auto iter = counters_.find(var_name);
  if(iter == counters_.end()){
    return nullptr;
  }
  return iter->second;
}


ReplicationCounter* ReplicationCounterManager::GetCounter(string var_name){
  mutex_lock l(mu_);
  auto iter = counters_.find(var_name);
  if(iter == counters_.end()){
    return nullptr;
  }
  return iter->second;
}


int ReplicationCounterManager::number_counters(){
  return counters_.size();
}

// The global replication counter manager.
ReplicationCounterManager g_replication_counter_manager;

} // end namespace tensorflow



