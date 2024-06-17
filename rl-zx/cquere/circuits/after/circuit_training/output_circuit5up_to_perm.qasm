OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
swap q[3], q[5];
swap q[1], q[3];
h q[4];
h q[0];
rz(1.5*pi) q[2];
h q[3];
h q[5];
h q[4];
h q[2];
cz q[1], q[5];
cz q[1], q[4];
cz q[1], q[3];
cz q[1], q[2];
h q[1];
cz q[1], q[3];
h q[3];
cx q[3], q[2];
cz q[1], q[5];
rz(0.49586776074068584*pi) q[1];
h q[5];
h q[1];
cz q[1], q[5];
rz(0.49586776074068584*pi) q[5];
h q[5];
cx q[5], q[2];
cz q[1], q[3];
rz(1.9958677607406858*pi) q[3];
h q[3];
cx q[3], q[2];
h q[1];
rz(0.49586776074068584*pi) q[1];
h q[1];
cx q[1], q[2];
h q[5];
rz(0.9958677607406858*pi) q[5];
h q[5];
h q[3];
cz q[3], q[5];
rz(1.9958677607406858*pi) q[3];
h q[3];
h q[1];
cz q[1], q[5];
cz q[1], q[3];
rz(1.4958677607406858*pi) q[1];
h q[4];
h q[1];
h q[5];
h q[3];
cz q[3], q[5];
cz q[1], q[5];
cz q[1], q[3];
rz(1.9958677607406858*pi) q[5];
rz(0.5*pi) q[3];
rz(1.0*pi) q[0];
h q[5];
h q[4];
h q[3];
h q[0];
cz q[4], q[5];
cz q[3], q[5];
cz q[2], q[5];
cz q[1], q[5];
rz(0.5*pi) q[5];
rz(1.0*pi) q[3];
h q[3];
h q[2];
rz(1.0*pi) q[1];
h q[1];
