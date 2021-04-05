
from scale_generator import ScaleGenerator,ratio_to_cents
import numpy as np
config=[
    {
        "name":"Minor 2nd",
        "matrix":np.array(
        [[0,0,1,0,0,0,0],
        [0,0,0,0,0,0,1]]
        ),
        "weight":1*np.ones(2),
        "weight_et":0.01*np.ones(2),
        "target":ratio_to_cents(16/15),
        "semis":1
    },
    {
        "name":"Major 2nd",
        "matrix":np.array(
            [[1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0]
            ]
        ),
        "weight":1*np.ones(5),
        "weight_et":0.01*np.ones(5),
        "target":ratio_to_cents(9/8),
        "semis":2
    },
    {
        "name":"Minor 3rd",
        "matrix":np.array(
            [[0,1,1,0,0,0,0],
            [0,0,1,1,0,0,0],
            [0,0,0,0,0,1,1],
            [1,0,0,0,0,0,1]
            ]
        ),
        "weight":2*np.ones(4),
        "weight_et":0.1*np.ones(4),
        "target":ratio_to_cents(6/5),
        "semis":3
    },
    {
        "name":"Major 3rd",
        "matrix":np.array(
            [[1,1,0,0,0,0,0],
            [0,0,0,1,1,0,0],
            [0,0,0,0,1,1,0]
            ]
        ),
        "weight":1.5*np.ones(3),
        "weight_et":0.1*np.ones(3),
        "target":ratio_to_cents(5/4),
        "semis":4
    },
    
    {
    "name":"Perfect 4th",
    "matrix":np.array([
        [1,1,1,0,0,0,0],
        [0,1,1,1,0,0,0],
        [0,0,1,1,1,0,0],
        [0,0,0,0,1,1,1],
        [1,0,0,0,0,1,1],
    ]),
    "weight":None,
    "weight_et":None,
    "target":ratio_to_cents(4/3),
    "semis":5
    },
    {
        "name":"Perfect 5th",
        "matrix":np.array([
            [1,1,1,1,0,0,0],
            [0,1,1,1,1,0,0],
            [0,0,1,1,1,1,0],
            [0,0,0,1,1,1,1],
            [1,0,0,0,1,1,1],
            [1,1,0,0,0,1,1]
        ]),
        "weight":4*np.ones(6),
        "weight_et":0.1*np.ones(6),
        "target":ratio_to_cents(3/2),
        "semis":7
    },
    {
        "name":"Octave",
        "matrix":np.ones((7,7)),
        "weight":None,
        "weight_et":None,
        "target":ratio_to_cents(2),
        "semis":12
    }
]


sg = ScaleGenerator(config,octave_locked=True,order=4,n_iter=100)
sg.optimize()
sg.print_report()
for i in range(0,12):
    for j in range(0,12):
        tun_strr = sg.export_tun('output_files/major/major_scale_optimized.tun',offset=i,in_tune_key=j,add_scale_to_filename=True)
