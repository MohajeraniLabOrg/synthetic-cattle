frames_to_render = [0, 10, 20, 30]

categories = [
    {
        "id": 1,
        "name": "cow",
        "supercategory": "",
        "keypoints": [
            "Back1",
            "Back2",
            "Back3",
            "Back4",
            "Head",
            "Nose",
            "Neck",
            "L_Shoulder",
            "L_Elbow",
            "L_F_Paw",
           "R_Shoulder",
            "R_Elbow",
            "R_F_Paw",
            "Belly",
            "L_Hip",
            "L_Knee",
            "L_H_Paw",
            "R_Hip",
           "R_Knee",
            "R_H_Paw"
        ],
        "skeleton": [
            [
                3,
                4
            ],
            [
                16,
                17
            ],
            [
                12,
                13
            ],
            [
                8,
                9
            ],
            [
                11,
                14
            ],
            [
                18,
                1
            ],
            [
                6,
                5
            ],
            [
                18,
                19
            ],
            [
                14,
                18
            ],
            [
                14,
                15
            ],
            [
                9,
                10
            ],
            [
                8,
                14
            ],
            [
                1,
                2
            ],
            [
                19,
                20
            ],
            [
                15,
                1
            ],
            [
                15,
                16
            ],
            [
                7,
                6
            ],
            [
                4,
                7
            ],
            [
                11,
                12
            ],
            [
                2,
                3
            ],
            [
                7,
                11
            ],
            [
                7,
                8
            ]
        ]
    }
]

img_template = {
    "id": 0,
    "width": 0,
    "height": 0,
    "file_name": "",
    "license": 0
}

ann_template = {
    "id": 0,
    "image_id": 0,
    "category_id": 1,
    "segmentation": [],
    "area": 0,
    "bbox": [],
    "iscrowd": 0,
    "attributes": {
        "occluded": False,
        "keyframe": False
    },
    "keypoints": [],
    "num_keypoints": 20
}


bones = [
    'bptr1-Back1',
    'bptr2-Back2',
    'bptr3-Back3',
    'bptr4-Back4',
    'bptr5-Head',
    'bptr6-Nose',
    'bptr7-Neck',
    'bptr8-L_Shoulder',
    'bptr9-L_Elbow',
    'bptr10-L_F_Paw',
    'bptr11-R_Shoulder',
    'bptr12-R_Elbow',
    'bptr13-R_F_Paw',
    'bptr14-Belly',
    'bptr15-L_Hip',
    'bptr16-L_Knee',
    'bptr17-L_H_Paw',
    'bptr18-R_Hip',
    'bptr19-R_Knee',
    'bptr20-R_H_Paw'
]
