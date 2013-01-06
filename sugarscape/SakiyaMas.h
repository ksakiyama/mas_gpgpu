#ifndef _SAKIYA_MAS_H_
#define _SAKIYA_MAS_H_

#if defined(_WIN32)||defined(_WIN64) /* Windows */
#define NOMINMAX
#endif

#include <iostream>
#include <cmath>

namespace mcl {

namespace Color {
	struct ColorElement {
		float value;
		float bias;
		ColorElement();
		ColorElement(float value_);
		ColorElement(float value_, float bias_);
	};

	static const float Red       = 0;
	static const float Orange    = 28;
	static const float Brown     = 27;
	static const float Yellow    = 60;
	static const float Green     = 105;
	static const float Lime      = 125;
	static const float Turquoise = 162;
	static const float Cyan      = 180;
	static const float Sky       = 200;
	static const float Blue      = 218;
  static const float Violet    = 271;
  static const float Magenta   = 326;
  static const float Pink      = 345;

  void transHSVtoRGB(float *dst, const float h, const float s, const float v);
  void trans(float *dst, float color, float bias);
};

namespace Random {
  void seed(unsigned int);
  unsigned int random();
  unsigned int random(unsigned int);
  unsigned int random(unsigned int, unsigned int);
  float randomf();
  double randomd();
};

};

namespace details {
  template <typename T>
  struct SetArgHandler {
    static ::size_t size(const T&) { return sizeof(T); }
    static T* ptr(T& value) { return &value; }
  };
}; /* end of namespace details */

#endif
