OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
ry(-pi/2) q[0];
ry(-pi/2) q[1];
rz(-7*pi/2) q[4];
rz(-3*pi/2) q[5];
rz(-3*pi/2) q[6];
ry(pi/2) q[9];
rx(pi) q[9];
rzz(pi/2) q[6],q[9];
ry(pi/2) q[6];
rz(-pi/2) q[9];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[6];
rz(-pi/2) q[6];
ry(-pi/2) q[6];
rz(-pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[6],q[9];
rz(-pi/2) q[6];
ry(pi/2) q[6];
rz(-pi) q[9];
rzz(pi/2) q[5],q[9];
ry(pi/2) q[5];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[5];
rz(-pi/2) q[5];
ry(-pi/2) q[5];
rz(-pi/2) q[9];
ry(pi/2) q[9];
rzz(pi/2) q[5],q[9];
rzz(pi/2) q[1],q[5];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rz(-pi/2) q[9];
rzz(pi/2) q[4],q[9];
rzz(pi/2) q[4],q[8];
rzz(pi/2) q[4],q[7];
rzz(pi/2) q[4],q[6];
rzz(pi/2) q[3],q[4];
rzz(pi/2) q[2],q[4];
rzz(pi/2) q[1],q[4];
rzz(pi/2) q[0],q[4];
rz(-pi) q[0];
rz(-pi/2) q[1];
ry(pi/2) q[1];
ry(-pi/2) q[4];
rz(-3.1170484460922063) q[4];
rzz(pi/2) q[0],q[4];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[1];
rz(-9.449322168266967) q[0];
rzz(pi/2) q[0],q[9];
rz(-3*pi/2) q[1];
rzz(pi/2) q[4],q[6];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rz(-3.1170484460922063) q[6];
rzz(pi/2) q[0],q[6];
ry(-pi/2) q[6];
rz(-3*pi) q[7];
rz(-3*pi) q[8];
rzz(pi/2) q[5],q[8];
rzz(pi/2) q[5],q[7];
ry(-pi/2) q[5];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[1];
rz(0.02454420749758679) q[9];
rzz(pi/2) q[6],q[9];
rz(pi) q[6];
ry(pi/2) q[9];
rzz(pi/2) q[4],q[9];
rzz(pi/2) q[0],q[4];
rzz(pi/2) q[0],q[9];
rzz(pi/2) q[0],q[6];
ry(pi/2) q[4];
rzz(pi/2) q[0],q[4];
ry(-pi/2) q[0];
rz(pi) q[4];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rz(-pi/2) q[5];
ry(pi/2) q[5];
rz(-1.5953405342924842) q[6];
rzz(pi/2) q[4],q[6];
rzz(pi/2) q[0],q[6];
ry(-pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[1];
ry(-pi/2) q[1];
rz(1.600752806797324) q[6];
ry(pi/2) q[9];
rzz(pi/2) q[0],q[9];
ry(-pi/2) q[0];
rz(1.5462521192973089) q[9];
ry(-pi/2) q[9];
rzz(pi/2) q[9],q[5];
rzz(pi/2) q[4],q[9];
rzz(pi/2) q[0],q[9];
rz(-9.449322168266967) q[0];
rz(-7.829437426476896) q[4];
rzz(pi/2) q[4],q[8];
rzz(pi/2) q[0],q[8];
rzz(pi/2) q[4],q[7];
rzz(pi/2) q[0],q[7];
rzz(pi/2) q[3],q[4];
rzz(pi/2) q[2],q[4];
rzz(pi/2) q[1],q[4];
rz(0.029956480002427455) q[1];
ry(pi/2) q[4];
rz(pi/2) q[5];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
rzz(pi/2) q[0],q[4];
rzz(pi/2) q[0],q[3];
rzz(pi/2) q[0],q[2];
rzz(pi/2) q[0],q[1];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[6];
rz(-1.6007528067973231) q[0];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[6];
rz(pi/2) q[0];
ry(pi) q[0];
ry(-pi/2) q[1];
rz(pi/2) q[4];
ry(pi/2) q[4];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[4];
rz(-9*pi/2) q[4];
rz(pi/2) q[6];
ry(-pi/2) q[6];
rzz(pi/2) q[6],q[9];
rz(-1.6007528067973222) q[6];
rzz(pi/2) q[6],q[8];
rzz(pi/2) q[6],q[7];
rzz(pi/2) q[0],q[6];
rz(pi/2) q[0];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[9];
rz(-6.25322882717716) q[0];
rzz(pi/2) q[0],q[8];
rzz(pi/2) q[0],q[7];
ry(pi/2) q[6];
rzz(pi/2) q[0],q[6];
rzz(pi/2) q[0],q[5];
ry(-pi/2) q[0];
rzz(pi/2) q[0],q[4];
rz(1.5901826342943792) q[0];
ry(-pi/2) q[0];
ry(pi/2) q[5];
rz(pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[0],q[6];
rz(-pi/2) q[0];
rzz(pi/2) q[0],q[4];
ry(-0.01938630749948255) q[0];
rz(-pi/2) q[0];
rzz(pi/2) q[0],q[6];
ry(0.019386307499481784) q[0];
rz(-pi) q[0];
rzz(pi/2) q[0],q[6];
rzz(pi/2) q[0],q[4];
ry(1.5901826342943783) q[0];
rz(-pi) q[0];
rzz(pi/2) q[1],q[6];
ry(0.029956480002426563) q[1];
rz(-pi/2) q[1];
rzz(pi/2) q[1],q[4];
ry(3.111636173587366) q[1];
rz(-pi/2) q[1];
rzz(pi/2) q[1],q[6];
ry(-3.111636173587367) q[1];
rz(-3*pi/2) q[1];
rzz(pi/2) q[1],q[4];
rzz(pi/2) q[1],q[5];
ry(pi/2) q[1];
ry(1.551410019295414) q[5];
rz(pi/2) q[5];
rzz(pi/2) q[5],q[4];
ry(-0.01938630749948255) q[5];
rz(-pi/2) q[5];
rzz(pi/2) q[5],q[6];
ry(0.019386307499481784) q[5];
rz(-pi) q[5];
rzz(pi/2) q[5],q[4];
rzz(pi/2) q[0],q[4];
ry(-pi/2) q[4];
ry(pi/2) q[5];
rzz(pi/2) q[0],q[5];
rz(-1.551410019295414) q[5];
ry(-pi/2) q[5];
rzz(pi/2) q[5],q[1];
rz(-pi/2) q[1];
ry(-pi/2) q[1];
rzz(pi/2) q[3],q[5];
rzz(pi/2) q[2],q[5];
ry(pi/2) q[6];
rzz(pi/2) q[5],q[6];
rzz(pi/2) q[0],q[5];
ry(-pi/2) q[0];
rzz(pi/2) q[4],q[5];
rz(-pi/2) q[4];
ry(pi/2) q[4];
rz(pi/2) q[6];
ry(pi/2) q[6];
rzz(pi/2) q[1],q[6];
rz(pi) q[1];
rz(-pi/2) q[6];
rzz(pi/2) q[0],q[6];
rz(-pi) q[0];
rzz(pi/2) q[0],q[5];
rzz(pi/2) q[0],q[1];
ry(pi/2) q[1];
rx(-pi/2) q[9];
