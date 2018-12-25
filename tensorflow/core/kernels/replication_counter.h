#ifndef TENSORFLOW_CORE_KERNELS_REPLICATION_COUNTER_H_
#define TENSORFLOW_CORE_KERNELS_REPLICATION_COUNTER_H_

#include <set>
#include <map>

#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class ReplicationCounter{
 public:
  explicit ReplicationCounter(string var_name)
           : var_name(var_name),
             pull_counter(0),
             send_counter(0){

    pull_worker_set.clear();
  }
  // Reset the counter.
  void ResetUnlock();  

  // The mutex.
  mutex mu;
  // The name of variable
  const string var_name;
  // // The last global step when send replication
  // int last_global_step;
  // The number of pull requests from workers.
  // This number maybe not equal to the size of pull_worker_set_.
  int pull_counter;
  // The number of send replication requests.
  // We accumulate the number of send replication requests, and 
  // the really send it after reaching a certain threshold.
  int send_counter;
  // The collection of workers who request pull this variable.
  std::set<string> pull_worker_set;

};

class ReplicationCounterManager{
 public:
  ReplicationCounterManager() {}

  ReplicationCounter* GetOrCreateCounter(string var_name);

  // Get the number of counters.
  int number_counters();

 private:
  mutex mu_;
  std::map<string, ReplicationCounter> counters_;
};

extern ReplicationCounterManager g_replication_counter_manager;

} // end namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_COUNTER_H_