#include "VecR4.h"

#include <iostream>
#include "PointR4.h"

VecR4::VecR4() {
	z0 = 0.0; z1 = 0.0;
	z2 = 0.0; z3 = 0.0;
}


VecR4::~VecR4() {

}

VecR4::VecR4(VecR4 & inp) {
	z1 = inp.getX(); z2 = inp.getY();
	z0 = inp.getT(); z3 = inp.getZ();
}

VecR4 & VecR4::operator=(VecR4 & rhs) {
	this->z1 = rhs.getX(); this->z2 = rhs.getY();
	this->z0 = rhs.getT(); this->z3 = rhs.getZ();
	return *this;
}

VecR4::VecR4(VecR4 && inp) {
	z1 = inp.getX(); z2 = inp.getY();
	z0 = inp.getT(); z3 = inp.getZ();
}

VecR4 & VecR4::operator=(VecR4 && rhs) {
	this->z1 = rhs.getX(); this->z2 = rhs.getY();
	this->z0 = rhs.getT(); this->z3 = rhs.getZ();
	return *this;
}

VecR4::VecR4(double T, double X, double Y, double Z) {
	z0 = T; z1 = X;
	z2 = Y; z3 = Z;
}

std::ostream &operator<<(std::ostream &os, VecR4 const &rhs) {
	return os << rhs.z0 << "*e_0 + " << rhs.z1 << "*e_1 + " << rhs.z2 << "*e_2 + " << rhs.z3 << "*e_3";
}

void VecR4::setT(double T) {
	z0 = T;
}

void VecR4::setX(double X) {
	z1 = X;
}

void VecR4::setY(double Y) {
	z2 = Y;
}

void VecR4::setZ(double Z) {
	z3 = Z;
}

double VecR4::getT() {
	return z0;
}

double VecR4::getX() {
	return z1;
}

double VecR4::getY() {
	return z2;
}

double VecR4::getZ() {
	return z3;
}
