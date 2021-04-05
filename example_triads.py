
from scale_generator_triads import ScaleGeneratorTriads,ratio_to_cents
import numpy as np
config=[
    {
        "name":"Minor 2nd",
        "matrix":np.array(
        [[0,0,1,0,0,0,0],
        [0,0,0,0,0,0,1]]
        ),
        "weight":None,
        "weight_et":None,
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
        "weight":None,
        "weight_et":None,
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
        "weight":None,#np.ones(4)*0.1,
        "weight_et":None,
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
        "weight":None,#np.ones(3),
        "weight_et":None,
        "target":ratio_to_cents(5/4),
        "semis":4
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
        "weight":None,#np.ones(6),
        "weight_et":None,
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


sg = ScaleGeneratorTriads(config,octave_locked=True,order=2,n_iter=100)
sg.optimize()
sg.print_report()
for i in range(0,12):
    for j in range(0,12):
        tun_strr = sg.export_tun('output_files/triads/major_scale_triad_optimized.tun',offset=i,in_tune_key=j,add_scale_to_filename=True)
