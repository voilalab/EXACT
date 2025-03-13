import numpy as np

###
# This script is used to interpolate the NIST absorption coefficients for the materials to
# match the energies used in the simulation. It can be easily adapted to other materials.
###

energies = np.arange(1,100,2)

# sanity check for PMMA
reported_energies = np.array([\
    1.00E+00, 1.50E+00, 2.00E+00, 3.00E+00, 4.00E+00,\
    5.00E+00, 6.00E+00, 8.00E+00, 1.00E+01, 1.50E+01,\
    2.00E+01, 3.00E+01, 4.00E+01, 5.00E+01, 6.00E+01,\
    8.00E+01, 1.00E+02, 1.50E+02, 2.00E+02, 3.00E+02,\
    4.00E+02, 5.00E+02, 6.00E+02, 8.00E+02, 1.00E+03,\
    1.25E+03, 1.50E+03, 2.00E+03, 3.00E+03, 4.00E+03,\
    5.00E+03, 6.00E+03, 8.00E+03, 1.00E+04, 1.50E+04, 2.00E+04])

mu_per_density_PMMA = np.array([\
    2.794E+03, 9.153E+02, 4.037E+02, 1.236E+02, 5.247E+01,\
    2.681E+01, 1.545E+01, 6.494E+00, 3.357E+00, 1.101E+00,\
    5.714E-01, 3.032E-01, 2.350E-01, 2.074E-01, 1.924E-01,\
    1.751E-01, 1.641E-01, 1.456E-01, 1.328E-01, 1.152E-01,\
    1.031E-01, 9.410E-02, 8.701E-02, 7.641E-02, 6.870E-02,\
    6.143E-02, 5.591E-02, 4.796E-02, 3.844E-02, 3.286E-02,\
    2.919E-02, 2.659E-02, 2.317E-02, 2.105E-02, 1.820E-02, 1.684E-02])

result = np.exp(np.interp(energies, reported_energies, np.log(mu_per_density_PMMA), \
                            left = -np.inf, right = -np.inf))


def splitstr(input):
    l = input.split('\n')
    return np.array([float(number) for number in l])

air_energy_str = '''1.000E+00
1.500E+00
2.000E+00
3.000E+00
3.203E+00
3.203E+00
4.000E+00
5.000E+00
6.000E+00
8.000E+00
1.000E+01
1.500E+01
2.000E+01
3.000E+01
4.000E+01
5.000E+01
6.000E+01
8.000E+01
1.000E+02
1.500E+02
2.000E+02
3.000E+02
4.000E+02
5.000E+02
6.000E+02
8.000E+02
1.000E+03
1.250E+03
1.500E+03
2.000E+03
3.000E+03
4.000E+03
5.000E+03
6.000E+03
8.000E+03
1.000E+04
1.500E+04
2.000E+04'''

air_mu_over_rho_str = '''3.606E+03
1.191E+03
5.279E+02
1.625E+02
1.340E+02
1.485E+02
7.788E+01
4.027E+01
2.341E+01
9.921E+00
5.120E+00
1.614E+00
7.779E-01
3.538E-01
2.485E-01
2.080E-01
1.875E-01
1.662E-01
1.541E-01
1.356E-01
1.233E-01
1.067E-01
9.549E-02
8.712E-02
8.055E-02
7.074E-02
6.358E-02
5.687E-02
5.175E-02
4.447E-02
3.581E-02
3.079E-02
2.751E-02
2.522E-02
2.225E-02
2.045E-02
1.810E-02
1.705E-02'''

air_energies = splitstr(air_energy_str)
air_mu_over_rho = splitstr(air_mu_over_rho_str)
air_result = np.exp(np.interp(energies, air_energies, np.log(air_mu_over_rho), \
                            left = -np.inf, right = -np.inf))

print(air_result)

water_energy_str = '''1.00E+00
1.50E+00
2.00E+00
3.00E+00
4.00E+00
5.00E+00
6.00E+00
8.00E+00
1.00E+01
1.50E+01
2.00E+01
3.00E+01
4.00E+01
5.00E+01
6.00E+01
8.00E+01
1.00E+02
1.50E+02
2.00E+02
3.00E+02
4.00E+02
5.00E+02
6.00E+02
8.00E+02
1.00E+03
1.25E+03
1.50E+03
2.00E+03
3.00E+03
4.00E+03
5.00E+03
6.00E+03
8.00E+03
1.00E+04
1.50E+04
2.00E+04'''

water_mu_over_rho_str = '''4.078E+03
1.376E+03
6.173E+02
1.929E+02
8.278E+01
4.258E+01
2.464E+01
1.037E+01
5.329E+00
1.673E+00
8.096E-01
3.756E-01
2.683E-01
2.269E-01
2.059E-01
1.837E-01
1.707E-01
1.505E-01
1.370E-01
1.186E-01
1.061E-01
9.687E-02
8.956E-02
7.865E-02
7.072E-02
6.323E-02
5.754E-02
4.942E-02
3.969E-02
3.403E-02
3.031E-02
2.770E-02
2.429E-02
2.219E-02
1.941E-02
1.813E-02'''

water_energies = splitstr(water_energy_str)
water_mu_over_rho = splitstr(water_mu_over_rho_str)
water_result = np.exp(np.interp(energies, water_energies, np.log(water_mu_over_rho), \
                            left = -np.inf, right = -np.inf))

print(water_result)

