OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
cx q[2], q[1];
rz(1.5*pi) q[3];
cx q[2], q[3];
cx q[3], q[1];
rz(0.5*pi) q[3];
cz q[2], q[4];
cx q[2], q[5];
cx q[3], q[5];
rz(-0.5*pi) q[5];
rz(1.5*pi) q[1];
h q[2];
rz(1.0041322392593142*pi) q[2];
cx q[2], q[5];
cx q[5], q[1];
rz(0.5041322392593142*pi) q[5];
cx q[3], q[5];
rz(0.5041322392593142*pi) q[5];
cx q[2], q[5];
rz(0.5*pi) q[5];
rz(1.0041322392593142*pi) q[1];
cx q[3], q[2];
cx q[1], q[5];
rz(0.5041322392593142*pi) q[5];
cx q[3], q[5];
rz(0.5041322392593142*pi) q[5];
rz(0.00413223925931417*pi) q[2];
cx q[3], q[1];
cx q[5], q[2];
rz(0.00413223925931417*pi) q[1];
cx q[5], q[1];
cx q[4], q[5];
h q[5];
cx q[5], q[3];
cx q[5], q[2];
cx q[5], q[1];
rz(1.5*pi) q[5];
x q[0];
x q[3];
x q[1];
