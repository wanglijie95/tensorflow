#ifndef TENSORFLOW_FRAMEWORK_SHADOW_VAR_H_
#define TENSORFLOW_FRAMEWORK_SHADOW_VAR_H_


#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
class ShadowVar{
  public:
    explicit ShadowVar(int64 global_step, string name, const Tensor& tensor)
             : global_step_(global_step),
               name_(name), 
               tensor_(tensor) {}

    /// Copy constructor.
    ShadowVar(const ShadowVar& other)
             : global_step_(other.global_step()),
               name_(other.name()), 
               tensor_(other.val()) {}

    const int64 global_step() const {return global_step_; }   
    const string name() const {return name_; }
    const Tensor& val() const {return tensor_; }

    // Update only update the global_step and tensor
    void Update(int64 global_step, string name, const Tensor& tensor){
      // Check name
      CHECK_EQ(name_, name);
      // Check shape
      const bool same_shape = tensor_.shape().IsSameSize(tensor.shape());
      CHECK(same_shape);
      // Update assign
      global_step_ = global_step;
      tensor_ = tensor;
    }

  private:
    int64 global_step_;
    string name_;
    Tensor tensor_;
};

class ShadowManager{
public:

  ShadowManager() { }

  //Some functions for shadows_ (insert shadow, delete shadow, get_shadow)
  //When insert, If shadow name is same, the old shadow will be replaced.
  void InsertShadow(int64 global_step, string name, const Tensor& tensor);

  // Get a shadow named `name`
  const ShadowVar GetShadow(string name);

  // Get all shadows' name in this worker.
  void GetAllShadowNames(std::vector<string>* all_shadow_names,
                         std::vector<int64>* all_shadow_steps);

  // Get the number of shadows.
  int number_shadows();

  // Get the global step;
  int64 global_step();

private:
  mutex mu_;
  std::map<string, ShadowVar> shadows_;
  int64 global_step_ = 0;
};

// The definition of `g_shadow_manager` is in "tensorflow/core/framework/shadow_op.cc"
// avoid `multiple definition` error
extern ShadowManager g_shadow_manager;

} // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_SHADOW_VAR_H_