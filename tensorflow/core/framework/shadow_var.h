#ifndef TENSORFLOW_FRAMEWORK_SHADOW_VAR_H_
#define TENSORFLOW_FRAMEWORK_SHADOW_VAR_H_


#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {
class ShadowVar{
  public:
    explicit ShadowVar(int64 global_step, string name, const Tensor& tensor)
             : global_step_(global_step),
               name_(name), 
               tensor_(tensor) {}
    
    int64 global_step() {return global_step_; }   
    string name() {return name_; }
    const Tensor& val() {return tensor_; }
  
  private:
    mutex mu_;
    int64 global_step_;
    string name_;
    const Tensor tensor_;
};

class ShadowManager{
public:

  ShadowManager() { }

  //Some functions for shadows_ (insert shadow, delete shadow, get_shadow)
  //When insert, If shadow name is same, the old shadow will be replaced.
  void InsertShadow(ShadowVar* shadow);
  void InsertShadow(int64 global_step, string name, const Tensor& tensor);
  void DeleteShadow(string name);

  // This function is used by `InsertShadow` and `DeleteShadow`.
  // When call this function, the lock is acquired, so we don't
  // need to acquire the lock agagin. Otherwise it causes a deadlock.
  ShadowVar* GetShadowUnlock(string name);

  // Get a shadow named `name`
  ShadowVar* GetShadow(string name);

  // Get all shadows' name in this worker.
  void GetAllShadowNames(std::vector<string>* all_shadow_names,
                         std::vector<int64>* all_shadow_steps);

  // Get the number of shadows.
  int number_shadows();

private:
  mutex mu_;
  std::map<string, ShadowVar*> shadows_;
};

// The definition of `g_shadow_manager` is in "tensorflow/core/framework/shadow_op.cc"
// avoid `multiple definition` error
extern ShadowManager g_shadow_manager;

} // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_SHADOW_VAR_H_