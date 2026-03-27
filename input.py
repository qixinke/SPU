#!/usr/bin/env python
# coding: utf-8
# Author: Qxk
# Time:2024年7月26日17:56:30

import puss
from puss import reaction_bools
from puss import single_reaction_bool
from puss import prepolymer_reaction
from puss import prepolymer_reaction_bools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.model_selection import train_test_split  #分割数据集
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error   #模型评价
import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
###
polyol= '[Re]C(COC(C)COC(COC(COC(COC(COC(COC(COC(COC(C)COC(C)COC(COC(COC(COC(COC(COC(COC(C[Re])C)C)C)C)C)C)C)C)C)C)C)C)C)C)C'
Diisocyanate ='O=C=NCCCCCCN=C=O'
extender1 = '[Rn]C1=CC=C(CCC2=CC=C([Rn])C=C2)C=C1'
extender2 = '[Rn]C1=CC=C(SSC2=CC=C([Rn])C=C2)C=C1'
extender12 = 'O=C=NCCCCCCN=C=O'
extender11='O=C=NCCCCCCN=C=O'
pupiece=single_reaction_bool(polyol,Diisocyanate,extender1,extender2)
polyol1=['[Re]C(COC(C)COC(COC(COC(COC(COC(COC(COC(COC(C)COC(C)COC(COC(COC(COC(COC(COC(COC(C[Re])C)C)C)C)C)C)C)C)C)C)C)C)C)C)']
Diisocyanate1 = ['O=C=NCCCCCCN=C=O']
prepolymer=prepolymer_reaction(polyol1,Diisocyanate1)
Diisocyanate_ratio = 0.5
polyol_ratio=0.25
extender2_ratio=0.125
extender1_ratio=0.125
polyol_MW=1000
mac_rate=100
self_time=10
self_tem=100



mol = Chem.MolFromSmiles(Diisocyanate)
di_mol_weight = Descriptors.MolWt(mol)
mol = Chem.MolFromSmiles(extender11)
ex1_mol_weight = Descriptors.MolWt(mol)
mol = Chem.MolFromSmiles(extender12)
ex2_mol_weight = Descriptors.MolWt(mol)
pupiece1=pupiece[0]
mol = Chem.MolFromSmiles(pupiece1)
pupiece_mol_weight = Descriptors.MolWt(mol)
R=Diisocyanate_ratio/(polyol_ratio+extender2_ratio+extender1_ratio)
Hs_wt=(Diisocyanate_ratio*di_mol_weight+extender1_ratio*ex1_mol_weight+extender2_ratio*ex2_mol_weight)/(polyol_ratio*polyol_MW+Diisocyanate_ratio*di_mol_weight+extender1_ratio*ex1_mol_weight+extender2_ratio*ex2_mol_weight)
dict={'polyol':polyol,'Diisocyanate':Diisocyanate,'extender1':extender11,'extender2':extender12,'prepolymer':prepolymer,'pupiece':pupiece}
STR=pd.DataFrame(dict)
data={ 'Diisocyanate_ratio':[Diisocyanate_ratio],
       'polyol_ratio':polyol_ratio,
      'extender2_ratio':extender2_ratio,
      'extender1_ratio':extender1_ratio,
      'piece_mw':pupiece_mol_weight,
      'polyol_MW':polyol_MW,
       'R': R,
      'Hs_wt':Hs_wt,
      'mac_rate':mac_rate,
      'self_time':self_time,
      'self_tem':self_tem}
data=pd.DataFrame(data)
data1={ 'Diisocyanate_ratio':[Diisocyanate_ratio],
       'polyol_ratio':polyol_ratio,
      'extender2_ratio':extender2_ratio,
      'extender1_ratio':extender1_ratio,
      'piece_mw':pupiece_mol_weight,
      'polyol_MW':polyol_MW,
       'R': R,
      'Hs_wt':Hs_wt,
      'mac_rate':mac_rate,
      }
data1=pd.DataFrame(data)

##############################################
def  Ts_input():
    df=pd.DataFrame()
    smiles = STR.iloc[:,5]


    smiles = STR.iloc[:, 5]
    descriptor_names = ['SMR_VSA6',
                        'FpDensityMorgan2', 'BCUT2D_MRLOW',
                        'PEOE_VSA10', 'VSA_EState6', 'VSA_EState9',
                        'PEOE_VSA7',
                        'HallKierAlpha', 'SPS',
                        'BCUT2D_MWHI',
                        'MinAbsEStateIndex', 'MaxAbsEStateIndex',
                        'qed', 'BCUT2D_MRHI',
                        'BalabanJ',
                        'EState_VSA6', 'MaxPartialCharge', 'MinEStateIndex',
                        'PEOE_VSA9',
                        'NumAliphaticHeterocycles', 'VSA_EState5',
                        'VSA_EState2', 'BertzCT']
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    for index, smiles_i in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'piece' + descriptor
        descriptors1.append(descriptor)
    descriptors1 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors5 = pd.DataFrame(descriptors1)



    smiles = STR.iloc[:, 2]
    descriptor_names = ['PEOE_VSA2', 'VSA_EState4', 'VSA_EState7', 'PEOE_VSA9',
                        ]
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    for index, smiles_i in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'ex1' + descriptor
        descriptors1.append(descriptor)
    descriptors2 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors2 = pd.DataFrame(descriptors2)

    smiles = STR.iloc[:, 3]
    descriptor_names = ['PEOE_VSA9', 'BCUT2D_CHGHI', 'MinAbsEStateIndex', 'BCUT2D_LOGPLOW', 'FpDensityMorgan1', 'PEOE_VSA7',
                        'BCUT2D_LOGPHI', 'FpDensityMorgan3', 'CUT2D_LOGPLOW', 'pDensityMorgan1', 'EOE_VSA7', 'MolLogP',
                        'MaxPartialCharge', 'TPSA']
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'ex2' + descriptor
        descriptors1.append(descriptor)

    descriptors3 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors3 = pd.DataFrame(descriptors3)

    smiles = STR.iloc[:, 4]
    descriptor_names = ['BCUT2D_MWHI', 'SlogP_VSA5'
                        ]
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'pre' + descriptor
        descriptors1.append(descriptor)

    descriptors4 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors4 = pd.DataFrame(descriptors4)
    descriptors4

    descriptors=pd.concat([descriptors5,descriptors2,descriptors3,descriptors4,data1,df],axis=1)
    column_order = ['ex2PEOE_VSA9', 'R', 'ex2BCUT2D_CHGHI', 'pieceSMR_VSA6',
                    'pieceFpDensityMorgan2', 'preBCUT2D_MWHI', 'pieceBCUT2D_MRLOW',
                    'polyol_MW', 'piecePEOE_VSA10', 'pieceVSA_EState6', 'pieceVSA_EState9',
                    'preSlogP_VSA5', 'piecePEOE_VSA7', 'ex2MinAbsEStateIndex',
                    'ex1PEOE_VSA2', 'ex1VSA_EState4', 'piece_mw', 'ex2BCUT2D_LOGPHI',
                    'pieceHallKierAlpha', 'pieceSPS', 'polyol_ratio', 'ex2FpDensityMorgan3',
                    'ex1VSA_EState7', 'mac_rate', 'pieceBCUT2D_MWHI', 'extender1_ratio',
                    'pieceMinAbsEStateIndex', 'pieceMaxAbsEStateIndex', 'ex2BCUT2D_LOGPLOW',
                    'Diisocyanate_ratio', 'Hs_wt', 'pieceqed', 'pieceBCUT2D_MRHI',
                    'ex2FpDensityMorgan1', 'extender2_ratio', 'pieceBalabanJ',
                    'pieceEState_VSA6', 'pieceMaxPartialCharge', 'pieceMinEStateIndex',
                    'piecePEOE_VSA9', 'ex2PEOE_VSA7', 'ex2MolLogP',
                    'pieceNumAliphaticHeterocycles', 'ex1PEOE_VSA9', 'pieceVSA_EState5',
                    'pieceVSA_EState2', 'ex2MaxPartialCharge', 'pieceBertzCT', 'ex2TPSA']
    # 按照指定的列名重新索引DataFrame的列顺序
    ts = descriptors[column_order]
    return  ts


###########################################################


def Eb_input():
    df = pd.DataFrame()
    smiles = STR.iloc[:, 5]
    mw = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        mol_weight = Descriptors.MolWt(mol)
        mw.append(mol_weight)
    df['piece_mw'] = mw

    smiles = STR.iloc[:, 5]
    descriptor_names = [
        'BCUT2D_MWLOW', 'MinPartialCharge', 'VSA_EState3', 'SlogP_VSA5',
        'MolLogP', 'VSA_EState10', 'PEOE_VSA14',
        'PEOE_VSA8',
        'FpDensityMorgan3',
        'BCUT2D_MRHI', 'PEOE_VSA10', 'MaxAbsPartialCharge',
        'PEOE_VSA2', 'MaxEStateIndex',
        'EState_VSA9', 'FpDensityMorgan1',
        'PEOE_VSA7', 'qed',
        'VSA_EState7', 'Chi4v',
        'VSA_EState9', 'MinAbsEStateIndex',
        'EState_VSA5',
        'MinEStateIndex', 'BCUT2D_MWHI',
        'VSA_EState1', 'HallKierAlpha']

    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'piece' + descriptor
        descriptors1.append(descriptor)
    descriptors1 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors5 = pd.DataFrame(descriptors1)

    smiles = STR.iloc[:, 2]
    descriptor_names = ['SlogP_VSA5', 'Chi2v', 'Chi4v', 'Kappa3', 'HallKierAlpha', 'HeavyAtomMolWt', 'EState_VSA6',
                        'VSA_EState10', 'EState_VSA8'
                        ]

    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    for index, smiles_i in enumerate(smiles):
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'ex1' + descriptor
        descriptors1.append(descriptor)

    descriptors2 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors2 = pd.DataFrame(descriptors2)

    # dataset=pd.read_csv('test.csv')
    smiles = STR.iloc[:, 3]
    descriptor_names = ['PEOE_VSA7', 'HallKierAlpha', 'EState_VSA7', 'PEOE_VSA14', 'Chi4v', 'Kappa3']

    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    for index, smiles_i in enumerate(smiles):
        # print(index + 1, '/' ,len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'ex2' + descriptor
        descriptors1.append(descriptor)

    descriptors3 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors3 = pd.DataFrame(descriptors3)

    smiles = STR.iloc[:, 4]
    descriptor_names = ['NumRotatableBonds', 'BalabanJ', 'qed'
                        ]
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        # print(index + 1, '/' ,len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'pre' + descriptor
        descriptors1.append(descriptor)

    descriptors4 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors4 = pd.DataFrame(descriptors4)

    descriptors = pd.concat([descriptors5, descriptors2, descriptors3, descriptors4, data1, df], axis=1)
    column_order = ['preNumRotatableBonds', 'ex1SlogP_VSA5',
                    'pieceBCUT2D_MWLOW', 'pieceMinPartialCharge', 'ex1Chi2v',
                    'ex2PEOE_VSA7', 'pieceVSA_EState3', 'R', 'pieceSlogP_VSA5', 'ex1Chi4v',
                    'pieceMolLogP', 'pieceVSA_EState10', 'piecePEOE_VSA14',
                    'ex2HallKierAlpha', 'ex2EState_VSA7', 'ex1Kappa3', 'piecePEOE_VSA8',
                    'Diisocyanate_ratio', 'pieceFpDensityMorgan3', 'ex1HallKierAlpha',
                    'pieceBCUT2D_MRHI', 'piecePEOE_VSA10', 'pieceMaxAbsPartialCharge',
                    'piecePEOE_VSA2', 'ex2PEOE_VSA14', 'pieceMaxEStateIndex',
                    'pieceEState_VSA9', 'pieceFpDensityMorgan1', 'preBalabanJ',
                    'ex1HeavyAtomMolWt', 'piecePEOE_VSA7', 'pieceqed', 'extender2_ratio',
                    'pieceVSA_EState7', 'pieceChi4v', 'ex2Chi4v', 'ex2Kappa3',
                    'ex1EState_VSA6', 'pieceVSA_EState9', 'pieceMinAbsEStateIndex',
                    'pieceEState_VSA5', 'polyol_ratio', 'preqed', 'ex1VSA_EState10',
                    'pieceMinEStateIndex', 'ex1EState_VSA8', 'Hs_wt', 'pieceBCUT2D_MWHI',
                    'mac_rate', 'pieceVSA_EState1', 'extender1_ratio', 'pieceHallKierAlpha',
                    ]

    Eb = descriptors[column_order]

    return  Eb


def E_input() -> object:
    df = pd.DataFrame()
    smiles = STR.iloc[:, 3]
    mw = []
    for smile in smiles:
        mol = Chem.MolFromSmiles(smile)
        # 计算分子的相对分子质量
        mol_weight = Descriptors.MolWt(mol)
        mw.append(mol_weight)
    df['ex2_mw'] = mw
    # dataset=pd.read_csv('test.csv')
    smiles = STR.iloc[:, 5]
    descriptor_names = ['VSA_EState6', 'BCUT2D_MRHI', 'EState_VSA6', 'PEOE_VSA10', 'fr_C_O_noCOO', 'SlogP_VSA5',
                        'BertzCT',
                        'EState_VSA1', 'SMR_VSA5', 'VSA_EState4', 'PEOE_VSA7', 'FpDensityMorgan2'
        , 'EState_VSA2', 'BCUT2D_CHGLO', 'HallKierAlpha', 'BCUT2D_LOGPHI', 'MaxEStateIndex', 'BCUT2D_MWHI', 'Chi4n',
                        'SlogP_VSA3', 'VSA_EState3', 'VSA_EState9', 'VSA_EState2', 'MinAbsEStateIndex', 'Chi4v',
                        'NumHeteroatoms', 'Chi3v', 'Kappa3', 'MinEStateIndex', 'PEOE_VSA14', 'EState_VSA3',
                        'VSA_EState5',
                        'PEOE_VSA8', 'BCUT2D_CHGHI'
                        ]
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        # print(index + 1, '/' ,len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'piece' + descriptor
        descriptors1.append(descriptor)

    descriptors1 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors5 = pd.DataFrame(descriptors1)
    smiles = STR.iloc[:, 2]
    descriptor_names = ['EState_VSA8',
                        'qed',
                        'SPS.1',
                        'PEOE_VSA8',
                        'EState_VSA4', 'BCUT2D_LOGPHI'
                        ]
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        # print(index + 1, '/' ,len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'ex1' + descriptor
        descriptors1.append(descriptor)

    descriptors2 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors2 = pd.DataFrame(descriptors2)

    smiles = STR.iloc[:, 3]
    descriptor_names = ['MolLogP',
                        'Chi4n',
                        'MinAbsEStateIndex',
                        'BCUT2D_LOGPHI']
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        # print(index + 1, '/' ,len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'ex2' + descriptor
        descriptors1.append(descriptor)

    descriptors3 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors3 = pd.DataFrame(descriptors3)

    # dataset=pd.read_csv('test.csv')
    smiles = STR.iloc[:, 4]
    descriptor_names = ['Chi1n', 'Chi4v'
        , 'MinEStateIndex',
                        'Chi4n',
                        'SMR_VSA6',
                        'Kappa3',
                        'MaxPartialCharge',
                        'PEOE_VSA8'
                        ]
    descriptors = []
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    for index, smiles_i in enumerate(smiles):
        # print(index + 1, '/' ,len(smiles))
        molecule = Chem.MolFromSmiles(smiles_i)
        descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
    descriptors1 = []
    for descriptor in descriptor_names:
        descriptor = 'pre' + descriptor
        descriptors1.append(descriptor)

    descriptors4 = pd.DataFrame(descriptors, columns=descriptors1)
    descriptors4 = pd.DataFrame(descriptors4)

    descriptors = pd.concat([descriptors5, descriptors2, descriptors3, descriptors4, data, df], axis=1)

    column_order = ['pieceVSA_EState6', 'pieceBCUT2D_MRHI', 'ex1EState_VSA8',
                    'pieceEState_VSA6', 'piecePEOE_VSA10', 'piecefr_C_O_noCOO',
                    'pieceSlogP_VSA5', 'pieceBertzCT', 'pieceEState_VSA1', 'ex1qed',
                    'preChi1n', 'pieceSMR_VSA5', 'ex2MolLogP', 'ex1BCUT2D_LOGPHI',
                    'pieceVSA_EState4', 'preChi4v', 'piecePEOE_VSA7', 'ex2Chi4n',
                    'pieceFpDensityMorgan2', 'ex1SPS.1', 'preMinEStateIndex',
                    'pieceEState_VSA2', 'pieceBCUT2D_CHGLO', 'pieceHallKierAlpha',
                    'pieceBCUT2D_LOGPHI', 'pieceMaxEStateIndex', 'pieceBCUT2D_MWHI',
                    'ex1PEOE_VSA8', 'ex2MinAbsEStateIndex', 'preChi4n', 'pieceChi4n',
                    'pieceSlogP_VSA3', 'preSMR_VSA6', 'ex1EState_VSA4', 'R',
                    'pieceVSA_EState3', 'ex2BCUT2D_LOGPHI', 'pieceVSA_EState9',
                    'pieceVSA_EState2', 'pieceMinAbsEStateIndex', 'pieceChi4v',
                    'pieceNumHeteroatoms', 'pieceChi3v', 'preKappa3', 'polyol_ratio',
                    'preMaxPartialCharge', 'pieceKappa3', 'Diisocyanate_ratio',
                    'pieceMinEStateIndex', 'piecePEOE_VSA14', 'pieceEState_VSA3',
                    'pieceVSA_EState5', 'piecePEOE_VSA8', 'pieceBCUT2D_CHGHI', 'Hs_wt',
                    'extender2_ratio', 'prePEOE_VSA8', 'extender1_ratio', 'self_time',
                    'self_tem']

    # 按照指定的列名重新索引DataFrame的列顺序
    E = descriptors[column_order]
    return E


