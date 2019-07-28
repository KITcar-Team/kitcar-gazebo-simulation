#include <gazebo/gazebo.hh>
#include <ignition/math.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>

namespace gazebo
{
  class AnimatedBox : public ModelPlugin
  {
    public: void Load(physics::ModelPtr _parent, sdf::ElementPtr sdf)
    {
      // Store the pointer to the model
      this->model = _parent;

        // create the animation
        gazebo::common::PoseAnimationPtr anim(
              new gazebo::common::PoseAnimation("test", 100.0, true));

        gazebo::common::PoseKeyFrame *key;

        for(sdf::ElementPtr el = sdf->GetFirstElement(); el; el = el->GetNextElement()) {
            float t = std::atof(el->GetAttribute("t")->GetAsString().c_str());
            float x = std::atof(el->GetAttribute("x")->GetAsString().c_str());
            float y = std::atof(el->GetAttribute("y")->GetAsString().c_str());
            float z = std::atof(el->GetAttribute("z")->GetAsString().c_str());
            float orientation = std::atof(el->GetAttribute("o")->GetAsString().c_str());

            key = anim->CreateKeyFrame(t);
            key->Translation(ignition::math::Vector3d(x, y, z));
            key->Rotation(ignition::math::Quaterniond(0, 0, orientation));
        }

        _parent->SetAnimation(anim);

    }

    // Pointer to the model
    private: physics::ModelPtr model;

    // Pointer to the update event connection
    private: event::ConnectionPtr updateConnection;
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(AnimatedBox)
}
