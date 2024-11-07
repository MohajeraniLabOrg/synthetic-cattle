dataset_info = dict(
    dataset_name='20kp',
    paper_info=dict(
        author='Ali Goldani',
        title='20kp',
        container='',
        year='',
        homepage='',
    ),
    keypoint_info={
        0:
        dict(
            name='Back1', 
            id=0,
            color=[135, 206, 250],
            type='upper',
            swap=''),
        1:
        dict(
            name='Back2',
            id=1,
            color=[100, 149, 237],
            type='upper',
            swap=''),
        2:
        dict(
            name='Back3',
            id=2,
            color=[65, 105, 225],
            type='upper',
            swap=''),
        3:
        dict(
            name='Back4',
            id=3,
            color=[0, 0, 255],
            type='upper',
            swap=''),
        4:
        dict(
            name='Head',
            id=4,
            color=[255, 247, 0],
            type='upper',
            swap=''),
        5:
        dict(
            name='Nose',
            id=5,
            color=[255, 223, 0],
            type='upper',
            swap=''),
        6:
        dict(
            name='Neck',
            id=6,
            color=[255, 255, 224],
            type='upper',
            swap=''),
        7:
        dict(
            name='L_Shoulder',
            id=7,
            color=[144, 238, 144],
            type='lower',
            swap='R_Shoulder'),
        8:
        dict(
            name='L_Elbow',
            id=8,
            color=[60, 179, 113],
            type='lower',
            swap='R_Elbow'),
        9:
        dict(
            name='L_F_Paw',
            id=9,
            color=[34, 139, 34],
            type='lower',
            swap='R_F_Paw'),
        10:
        dict(
            name='R_Shoulder',
            id=10,
            color=[152, 251, 152],
            type='lower',
            swap='L_Shoulder'),
        11:
        dict(
            name='R_Elbow',
            id=11,
            color=[50, 205, 50],
            type='lower',
            swap='L_Elbow'),
        12:
        dict(
            name='R_F_Paw',
            id=12,
            color=[0, 100, 0],
            type='lower',
            swap='R_F_Paw'),
        13:
        dict(
            name='Belly',
            id=13,
            color=[255, 165, 0],
            type='lower',
            swap=''),
        14:
        dict(
            name='L_Hip',
            id=14,
            color=[255, 160, 122],
            type='lower',
            swap='R_Hip'),
        15:
        dict(
            name='L_Knee',
            id=15,
            color=[250, 128, 114],
            type='lower',
            swap='R_Knee'),
        16:
        dict(
            name='L_H_Paw',
            id=16,
            color=[178, 34, 34],
            type='lower',
            swap='R_H_Paw'),
        17:
        dict(
            name='R_Hip',
            id=17,
            color=[255, 105, 97],
            type='lower',
            swap='L_Hip'),
        18:
        dict(
            name='R_Knee',
            id=18,
            color=[220, 20, 60],
            type='lower',
            swap='L_Knee'),
        19:
        dict(
            name='R_H_Paw',
            id=19,
            color=[139, 0, 0],
            type='lower',
            swap='L_H_Paw'),
    },
    skeleton_info={
        0: {'link': ('Back1', 'Back2'), 'id': 1, 'color': [121, 185, 225]},
        1: {'link': ('Back2', 'Back3'), 'id': 2, 'color': [90, 134, 213]},
        2: {'link': ('Back3', 'Back4'), 'id': 3, 'color': [58, 94, 202]},
        3: {'link': ('Back4', 'Neck'), 'id': 4, 'color': [0, 0, 229]},
        4: {'link': ('Head', 'Nose'), 'id': 5, 'color': [229, 222, 0]},
        5: {'link': ('Nose', 'Neck'), 'id': 6, 'color': [229, 200, 0]},
        6: {'link': ('Neck', 'L_Shoulder'), 'id': 7, 'color': [229, 229, 201]},
        7: {'link': ('L_Shoulder', 'L_Elbow'), 'id': 8, 'color': [129, 214, 129]},
        8: {'link': ('L_Elbow', 'L_F_Paw'), 'id': 9, 'color': [54, 161, 101]},
        9: {'link': ('Neck', 'R_Shoulder'), 'id': 10, 'color': [229, 229, 201]},
        10: {'link': ('R_Shoulder', 'R_Elbow'), 'id': 11, 'color': [136, 225, 136]},
        11: {'link': ('R_Elbow', 'R_F_Paw'), 'id': 12, 'color': [45, 184, 45]},
        12: {'link': ('L_Shoulder', 'Belly'), 'id': 13, 'color': [129, 214, 129]},
        13: {'link': ('R_Shoulder', 'Belly'), 'id': 14, 'color': [136, 225, 136]},
        14: {'link': ('Belly', 'L_Hip'), 'id': 15, 'color': [229, 148, 0]},
        15: {'link': ('L_Hip', 'L_Knee'), 'id': 16, 'color': [229, 144, 109]},
        16: {'link': ('L_Knee', 'L_H_Paw'), 'id': 17, 'color': [225, 115, 102]},
        17: {'link': ('Belly', 'R_Hip'), 'id': 18, 'color': [229, 148, 0]},
        18: {'link': ('R_Hip', 'R_Knee'), 'id': 19, 'color': [229, 94, 87]},
        19: {'link': ('R_Knee', 'R_H_Paw'), 'id': 20, 'color': [198, 18, 54]},
        20: {'link': ('L_Hip', 'Back1'), 'id': 21, 'color': [229, 144, 109]},
        21: {'link': ('R_Hip', 'Back1'), 'id': 22, 'color': [229, 94, 87]}    
    },
    joint_weights=[1.] * 20,
    sigmas=[.1] * 20
    )