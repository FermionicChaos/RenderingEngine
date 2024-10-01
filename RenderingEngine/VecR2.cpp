//#include "stdafx.h"
#include "stdud.h"
#include "VecR2.h"


VecR2::VecR2() {
	x = 0; y = 0;
}


VecR2::~VecR2() {}

VecR2::VecR2(const VecR2 & vec) {
	x = vec.x; y = vec.y;
}

VecR2 & VecR2::operator=(const VecR2 & rhs) {
	this->x = rhs.x; this->y = rhs.y;
	return *this;
}

VecR2::VecR2(VecR2 && vec) {
	x = vec.x; y = vec.y;
}

VecR2 & VecR2::operator=(VecR2 && rhs) {
	this->x = rhs.x; this->y = rhs.y;
	return *this;
}

VecR2::VecR2(double a, double b){
	x = a; y = b;
}

VecR2 VecR2::operator+(const VecR2& rhs) {
	return VecR2(x + rhs.x, y + rhs.y);
}

VecR2 VecR2::operator-(const VecR2& rhs) {
	return VecR2(x - rhs.x, y - rhs.y);
}

double VecR2::operator*(const VecR2& rhs) {
	return (x*rhs.x + y*rhs.y);
}

VecR2 VecR2::operator*(double rhs) {
	VecR2 temp(x*rhs,y*rhs);
	return temp;
}

std::ostream & operator<<(std::ostream & os, VecR2 const & rhs) {
	return os << rhs.x << "i + " << rhs.y << "j";
}

VecR2 operator*(double lhs, const VecR2 & rhs) {
	VecR2 temp(lhs*rhs.y, lhs*rhs.y);
	return VecR2();
}
