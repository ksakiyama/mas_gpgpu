#include "SakiyaMas.h"

namespace mcl {

/*
* namespace Color
*/
namespace Color {
	ColorElement::ColorElement() {}
	ColorElement::ColorElement(float value_) : value(value_), bias(0) {}
	ColorElement::ColorElement(float value_, float bias_) : value(value_), bias(bias_) {}

	void transHSVtoRGB(float *dst, const float h, const float s, const float v) {
		if (0.f == s) {
			dst[0] = dst[1] = dst[2] = v;
			return;
		}

		int Hi = (int)floor(h / 60.f) % 6;
		float f = h / 60.f - (float)Hi;
		float p = v * (1.f - s);
		float q = v * (1.f - s * f);
		float t = v * (1.f - (1.f - f) * s);

		if (Hi == 0) {
			dst[0] = v, dst[1] = t, dst[2] = p;
		} else if (Hi == 1) {
			dst[0] = q, dst[1] = v, dst[2] = p;
		} else if (Hi == 2) {
			dst[0] = p, dst[1] = v, dst[2] = t;
		} else if (Hi == 3) {
			dst[0] = p, dst[1] = q, dst[2] = v;
		} else if (Hi == 4) {
			dst[0] = t, dst[1] = p, dst[2] = v;
		} else {
			dst[0] = v, dst[1] = p, dst[2] = q;
		}
	}

	void trans(float *dst, float color, float bias) {
		float hsv_h = color;
		float hsv_s = 1.0f - fabs(bias);
		float hsv_v = std::min( (1.0f + bias), 1.0f );
		transHSVtoRGB(dst, hsv_h, hsv_s, hsv_v);
	}
}; /* end of namespace Color */


/*
* namespace Random
*/
namespace Random {
	unsigned int xors_x = 123456789;
	unsigned int xors_y = 362436069;
	unsigned int xors_z = 521288629;
	unsigned int xors_w = 88675123;

	void seed(unsigned int seed_) {
		xors_x = seed_;
		xors_y = 362436069;
		xors_z = 521288629;
		xors_w = 88675123;
	}

	unsigned int random() {
		unsigned int t;
		t = (xors_x ^ (xors_x << 11));
		xors_x = xors_y; xors_y = xors_z; xors_z = xors_w; 
		return xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8));
	}

	unsigned int random(unsigned int high) {
		unsigned int t;
		t = (xors_x ^ (xors_x << 11));
		xors_x = xors_y; xors_y = xors_z; xors_z = xors_w; 
		xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8));
		return (xors_w % (high));
	}

	unsigned int random(unsigned int low, unsigned int high) {
		unsigned int t;
		t = (xors_x ^ (xors_x << 11));
		xors_x = xors_y; xors_y = xors_z; xors_z = xors_w; 
		xors_w = (xors_w ^ (xors_w >> 19)) ^ (t ^ (t >> 8));
		return low + (xors_w % (high - low));
	}

	float randomf() {
		return ( random( 0, 1000000 ) / 1000000.0f );
	}

	double randomd() {
		return ( random( 0, 1000000 ) / 1000000.0 );
	}

}; /* end of namespace Random */

}; /* end of namespace mcl */
