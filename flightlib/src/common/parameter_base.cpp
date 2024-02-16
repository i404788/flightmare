#include "flightlib/common/parameter_base.hpp"
#include <iostream>
#include <fstream>

namespace flightlib {

ParameterBase::ParameterBase() {}

ParameterBase::ParameterBase(const json& cfg_node)
  : cfg_node_(cfg_node) {}

ParameterBase::ParameterBase(const std::string& cfg_path)
{
  std::ifstream f(cfg_path);
  cfg_node_ = json::parse(f);
}

ParameterBase::~ParameterBase() {}

}  // namespace flightlib
