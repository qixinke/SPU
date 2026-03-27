

import streamlit as st
import pymysql
import pandas as pd
import time
import streamlit as st
from streamlit_option_menu import option_menu
from puss import single_reaction_bool
from puss import prepolymer_reaction
from puss import prepolymer_reaction_bools
#import input
#from mysql.connector import Error
import matplotlib.pyplot as plt
import io
# 定义每个页面（标签页）的内容
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
def home_page():
    #st.header('SPU ')
    # 这里添加首页的内容
    st.markdown("""
    <hr style="border: 1px solid #ccc;" />
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # st.write('')
        image_path = 'img.png'  # 替换为你的图片文件路径

        # 显示图片
        st.image(image_path, use_column_width=True)
        #st.markdown('<span style="color:red">大bb记得晚上一起昏昏</span>', unsafe_allow_html=True)
    st.subheader('Self-healing polyurethane is an intelligent material capable of restoring some or all of its properties after damage. It integrates seamlessly across industries and biomedical fields due to its excellent mechanical, physical, and chemical properties, as well as good biocompatibility. When polyurethane experiences micro-cracks or structural damage, self-healing technology provides an effective solution. The advancement in self-healing polyurethane involves mechanisms based on dynamic covalent bonds, such as disulfide, Diels-Alder, and imine bonds, which allow for reversible changes and healing upon external stimuli. The evolution of self-healing polymers has reached the fourth generation, combining non-covalent and covalent bonds to achieve over 90% healing efficiency and significantly improved performance. The future of self-healing polymers may lie in multi-mechanism systems and the use of fillers or additives as healing agents, marking an innovative direction in material science for enhanced durability and sustainability.')
    st.markdown("""
        <hr style="border: 1px solid #ccc;" />
        """, unsafe_allow_html=True)
    st.markdown("""
    Kaifeng, Henan 475004, P.R. China

    COLLEGE OF CHEMISTRY AND MOLECULAR SCIENCES Henan University

    """, unsafe_allow_html=True)


def search_page():
    #st.title('Search')
    from streamlit_ketcher import st_ketcher
    search_value = st.text_input("Molecule", "")
    search_value = st_ketcher(search_value)
    #s#t.markdown(f"Smile code: ``{smile_code}``")
    #search_value = st.text_input(smile_code)
    #db_config = st.secrets["database"]

    if search_value:
        
        file='data.csv'
        df = pd.read_csv (file)
        row_text = df.astype(str).apply(
            lambda row: ' | '.join(row.values), 
            axis=1
        )
        
        mask = row_text.str.contains(
            search_value, 
            case=False,
            na=False,
            regex=False
        )
        
        df = df[mask]
        #db.close()
        if df.empty:
            st.subheader("NO FOUND")
        else:
            #df=df.head(10)
            df1 =df.iloc[:,8:]
            df=df.iloc[:,0:4]
            df=pd.concat([df,df1],axis=1)

            if 'row_index' not in st.session_state:
                st.session_state.row_index = 0

            # 创建一个滑块，允许用户选择行索引
            selected_row_index = st.slider('选择行索引:', min_value=0, max_value=len(df) - 1,
                                           value=st.session_state.row_index, step=1)

            # 更新session state
            st.session_state.row_index = selected_row_index

            # 选择特定的行
            selected_row = df.iloc[selected_row_index]

            # 显示所选行的数据
           # for col_name, col_value in selected_row.items():
                #st.write(f"{col_name}: {col_value}")
            button_container = st.columns(5)  # 第一个1表示第一列的宽度，0表示中间列（空列度
            spacer, spacer1, col1, col2, spacer2 = button_container
            # 创建一个按钮，允许用户查看下一行
            with col2:
                if st.button('Next Row'):
                    if st.session_state.row_index < len(df) - 1:
                        st.session_state.row_index += 1
                    else:
                        st.write("已经是最后一行了。")
            with col1:
                # 创建一个按钮，允许用户查看上一行
                if st.button('Previous Row'):
                    if st.session_state.row_index > 0:
                        st.session_state.row_index -= 1
                    else:
                        st.write("已经是第一行了。")

            # 创建一个按钮，允许用户查看上一行

            # 创建四个列来显示分子图像
            cols = st.columns(4)

            # 计数器，用于限制只显示前4个结构
            structure_count = 0

            # 显示所选行的索引
            #st.write(f"Data Row {selected_row_index + 1}")
            non_structure_data = []
            for col_name, col_value in selected_row.items():
                if isinstance(col_value, str) and structure_count < 4:  # 只处理字符串类型的列，并且限制数量
                    # 将SMILES字符串转换为分子对象
                    mol = Chem.MolFromSmiles(col_value)
                    if mol is not None:  # 确保SMILES字符串有效
                        # 将分子对象转换为图像
                        img = Draw.MolsToImage([mol], subImgSize=(200, 200))  # 设置图像大小

                        # 将图像保存到字节流中
                        image_stream = io.BytesIO()
                        img.save(image_stream, format='PNG')
                        image_stream.seek(0)  # 重置指针到流的开始位置

                        # 在Streamlit的相应列中显示图片
                        cols[structure_count].image(image_stream, caption=col_name, use_column_width=True)
                        structure_count += 1  # 增加结构计数器
                else:
                    #st.write(f"{col_name}: {col_value}")
                    non_structure_data.append((col_name, col_value))

                    # 显示非字符串类型的列，每行显示3个
            for i in range(0, len(non_structure_data), 3):
                # 创建3列
                cols = st.columns(3)
                for j, (col_name, col_value) in enumerate(non_structure_data[i:i + 3]):
                    cols[j].write(f"{col_name}{col_value}")
                    #else:

            # 如果查询结果为空，给出提示


def predict_page():
    #st.title('INFO')

    import joblib
    E_model = joblib.load('GBRE.joblib')
    Ts_model = joblib.load('GBRTs.joblib')
    Eb_model = joblib.load('GBREb.joblib')
    st.write("Monomer structure:")
    col0,col1, col2 = st.columns(3)
    with col0:
        polyol = st.text_input("polyol")
        Diisocyanate = st.text_input("Diisocyanate")
    # 在第一列中添加输入框
    with col1:
        extender1 = st.text_input("extender1")
        extender11 = st.text_input("extender11")
    # 在第二列中添加输入框
    with col2:
        extender2 = st.text_input("extender2")
        extender12 = st.text_input("extender12")
    st.write("Composition:")
    col3, col4, col5, col6,col7 = st.columns(5)
    with col3:
        polyol_ratio = st.number_input("polyol_ratio")
    with col4:
        Diisocyanate_ratio = st.number_input("Diisocyanate_ratio")
    with col5:
        extender1_ratio = st.number_input("extender1_ratio")
    with col6:
        extender2_ratio = st.number_input("extender2_ratio")
    with col7:
        polyol_MW = st.number_input("polyol_MW")
    st.write("Test conditions:")
    col8, col9, col10,col11,col12= st.columns(5)
    with col8:
        mac_rate = st.number_input("mac_rate")
    with col9:
        self_time = st.number_input("self_time")

    with col10:
        self_tem = st.number_input("self_tem")




    def Ts_input():
        df = pd.DataFrame()
        smiles = STR.iloc[:, 5]

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
        descriptor_names = ['PEOE_VSA9', 'BCUT2D_CHGHI', 'MinAbsEStateIndex', 'BCUT2D_LOGPLOW', 'FpDensityMorgan1',
                            'PEOE_VSA7',
                            'BCUT2D_LOGPHI', 'FpDensityMorgan3', 'CUT2D_LOGPLOW', 'pDensityMorgan1', 'EOE_VSA7',
                            'MolLogP',
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


        descriptors = pd.concat([descriptors5, descriptors2, descriptors3, descriptors4, data1, df], axis=1)
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
        return ts

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

        return Eb

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
            ,'EState_VSA2', 'BCUT2D_CHGLO', 'HallKierAlpha', 'BCUT2D_LOGPHI', 'MaxEStateIndex', 'BCUT2D_MWHI', 'Chi4n',
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


    #st.title("Software Developer  Prediction")
    #st.write("""### We need some information to predict the salary""")




    ok = st.button("Calculate")
    #if Diisocyanate & polyol
    try:
        if ok:

            mol = Chem.MolFromSmiles(Diisocyanate)
            di_mol_weight = Descriptors.MolWt(mol)

            mol = Chem.MolFromSmiles(extender11)
            ex1_mol_weight = Descriptors.MolWt(mol)

            mol = Chem.MolFromSmiles(extender12)
            ex2_mol_weight = Descriptors.MolWt(mol)

            pupiece = single_reaction_bool(polyol, Diisocyanate, extender1, extender2)

            polyol1 = [polyol]
            Diisocyanate1 = [Diisocyanate]
            prepolymer = prepolymer_reaction(polyol1, Diisocyanate1)

            pupiece1 = pupiece[0]
            mol = Chem.MolFromSmiles(pupiece1)
            pupiece_mol_weight = Descriptors.MolWt(mol)

            R = Diisocyanate_ratio / (polyol_ratio + extender2_ratio + extender1_ratio)
            Hs_wt = (Diisocyanate_ratio * di_mol_weight + extender1_ratio * ex1_mol_weight + extender2_ratio * ex2_mol_weight) /(polyol_ratio * polyol_MW + Diisocyanate_ratio * di_mol_weight + extender1_ratio * ex1_mol_weight + extender2_ratio * ex2_mol_weight)


            dict = {'polyol': polyol, 'Diisocyanate': Diisocyanate, 'extender1': extender11, 'extender2': extender12,
                    'prepolymer': prepolymer, 'pupiece': pupiece}
            STR = pd.DataFrame(dict)

            data = {'Diisocyanate_ratio': [Diisocyanate_ratio],
                    'polyol_ratio': polyol_ratio,
                    'extender2_ratio': extender2_ratio,
                    'extender1_ratio': extender1_ratio,
                    'piece_mw': pupiece_mol_weight,
                    'polyol_MW': polyol_MW,
                    'R': R,
                    'Hs_wt': Hs_wt,
                    'mac_rate': mac_rate,
                    'self_time': self_time,
                    'self_tem': self_tem}
            data = pd.DataFrame(data)

            data1 = {'Diisocyanate_ratio': [Diisocyanate_ratio],
                     'polyol_ratio': polyol_ratio,
                     'extender2_ratio': extender2_ratio,
                     'extender1_ratio': extender1_ratio,
                     'piece_mw': pupiece_mol_weight,
                     'polyol_MW': polyol_MW,
                     'R': R,
                     'Hs_wt': Hs_wt,
                     'mac_rate': mac_rate,
                     }
            data1 = pd.DataFrame(data1)


            E =E_input()
            Eb = Eb_input()
            Ts =Ts_input()
            Eb = Eb.values
            ypreEb = Eb_model.predict(Eb)
            ypreEb = 10 ** ypreEb
            Ts = Ts.values
            ypreTS = Ts_model.predict(Ts)
            ypreTS=10 **ypreTS
            E = E.values
            ypreE = E_model.predict(E)


            b = {'E':[ypreE], 'Eb':[ypreEb], 'TS':[ypreTS],'Hs_wt':[Hs_wt]}
            x = pd.DataFrame(b)

            smiles = pupiece
            descriptor_names = ['VSA_EState2', 'HallKierAlpha', 'BCUT2D_CHGHI']
            descriptors = []
            descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
            for index, smiles_i in enumerate(smiles):
                molecule = Chem.MolFromSmiles(smiles_i)
                descriptors.append(descriptor_calculator.CalcDescriptors(molecule))
            descriptors1 = []
            for descriptor in descriptor_names:
                descriptor = 'piece' + descriptor
                descriptors1.append(descriptor)
            descriptors2 = pd.DataFrame(descriptors, columns=descriptors1)
            descriptors3 = pd.DataFrame(descriptors2)

            out=pd.concat([x,descriptors3],axis=1)
            out

            fig, axs = plt.subplots(1, 3, figsize=(10, 3))  # 3行1列

            # 第一个子图
            x1 = x.iloc[:, 3]
            y1 = descriptors3.iloc[:, 0]
            axs[0].set_title('Ts')
            axs[0].set_xlabel('Hs_wt')
            axs[0].set_ylabel('pieceVSA_EState2')
            axs[0].axvline(x=0.35, color='r', linestyle='--', linewidth=1)
            axs[0].axhline(y=250, color='b', linestyle='-', linewidth=1)
            axs[0].plot(x1, y1, color='red', marker='o', markersize=8, linestyle='-', markerfacecolor='green',
                        markeredgewidth=2, markeredgecolor='blue')
            axs[0].set_ylim([0, 1300])
            axs[0].set_xlim(0, 0.6)

            # 第二个子图
            x2 = x.iloc[:, 3]
            y2 = descriptors3.iloc[:, 1]
            axs[1].set_title('Eb')
            axs[1].set_xlabel('Hs_wt')
            axs[1].set_ylabel('pieceHallKierAlpha')
            axs[1].axvline(x=0.35, color='r', linestyle='--', linewidth=1)
            axs[1].axhline(y=-30, color='b', linestyle='-', linewidth=1)
            axs[1].plot(x2, y2, color='red', marker='o', markersize=8, linestyle='-', markerfacecolor='green',
                        markeredgewidth=2, markeredgecolor='blue')
            axs[1].set_ylim([-80, 0])
            axs[1].set_xlim(0, 0.6)

            # 第三个子图
            x3 = x.iloc[:, 3]
            y3 = descriptors3.iloc[:, 2]
            axs[2].set_title('η')
            axs[2].set_xlabel('Hs_wt')
            axs[2].set_ylabel('pieceBCUT2D_CHGHI')
            axs[2].axvline(x=0.3, color='r', linestyle='--', linewidth=1)
            axs[2].axhline(y=2.75, color='b', linestyle='-', linewidth=1)
            axs[2].plot(x3, y3, color='red', marker='o', markersize=8, linestyle='-', markerfacecolor='green',
                        markeredgewidth=2, markeredgecolor='blue')
            axs[2].set_ylim([2.2, 3])
            axs[2].set_xlim(0, 0.6)

            # 调整子图间距
            plt.tight_layout()

            # 将matplotlib图表转换为Streamlit图表
            st.pyplot(fig)
    except Exception as e:
        # 出错时执行的代码
        a='error'
        a
def download_page():
    def get_data():
       
        file = 'data.csv'

        df = pd.read_csv(file)
        #df = pd.read_csv(data.csv)
        #db.close()
        return df

    # Streamlit界面
    #st.title('Display Data with Pagination')

    # 获取数据
    df = get_data()

    # 定义一个状态标志，用于跟踪程序是否已经运行过
    '''run_once = st.session_state.get("run_once", False)
    if not run_once:
        with st.spinner('Connecting to MySQL...'):
            time.sleep(3)
        st.session_state.run_once = True'''

    # 设置每页显示的行数
    rows_per_page = 10

    # 获取当前页码，默认为1
    current_page = st.session_state.get("current_page", 1)

    # 计算起始和结束索引
    start_index = (current_page - 1) * rows_per_page
    end_index = start_index + rows_per_page
    #df_display = df.iloc[start_index:end_index].reset_index(drop=True)
    #st.dataframe(df_display, use_container_width=True)

    # 分页显示数据
    st.dataframe(df.iloc[start_index:end_index],use_container_width=True)

    # 创建一个容器来居中显示按钮
    #button_container = st.columns(2)
    #col1, col2 = button_container
    button_container = st.columns(5)  # 第一个1表示第一列的宽度，0表示中间列（空列度
    spacer,spacer1,col1,col2,spacer2 = button_container


    # 在第二列中创建两个按钮并使用session_state来更新页码
    with col1:
        if st.button(' Prev ', key='prev'):
            new_page = max(current_page - 1, 1)  # 确保页码不会小于1
            st.session_state.current_page = new_page
    with col2:
        #st.write(" " * 2000)
        if st.button(' Next ', key=' Next '):
            new_page = current_page + 1
            st.session_state.current_page = new_page



    # 添加数据部分

           


def contact_page():
    st.title('Contact Us')
    st.write("Email: qixinke@henu.edu.cn")
    st.write("Address: Kaifeng, Henan 475004, P.R. China")

    # 主函数


import streamlit as st

# 模拟的用户凭据
USERNAME = "admin"
PASSWORD = "password"

USERNAME = "admin"
PASSWORD = "password"

import streamlit as st

# 模拟的用户凭据
USERNAME = "admin"
PASSWORD = "password"




USERNAME = "admin"
PASSWORD = "password"

#def login_page():
    #image_path = 'a.png'  # 替换为你的图片文件路径




    # 显示图片
    #st.image(image_path, use_column_width=True)

    # 创建一个登录表单
    #with st.form("login_form"):
       # st.write("登录到自修复聚氨酯数据库")
       # username = st.text_input("用户名")
       # password = st.text_input("密码", type="password")
       # submit_button = st.form_submit_button(label='用户登录')
    # 登录逻辑
       # if submit_button:
        #    if username == USERNAME and password == PASSWORD:
       #         st.success("登录成功！")
        #        st.session_state['logged_in'] = True  # 设置登录状态
         #   else:
          #      st.error("用户名或密码错误。", className="error-message")


# 运行登录页面


def main():
    #st.set_page_config(page_title='SPU', layout='wide')

    # 如果用户未登录，显示登录界面
   # if not st.session_state.get('logged_in', False):
       # login_page()
  #  else:
    st.title(' SPU')
    #st.write('continue to develop...')
    selected2 = option_menu(None, ["Home", "Search", "Download", 'Predict', 'Contact'],
                            icons=['house', 'search', "download", 'robot', 'envelope'],
                            menu_icon="cast", default_index=0, orientation="horizontal")

    if selected2 == "Home":
        home_page()
    elif selected2 == "Search":
        search_page()
    elif selected2 == "Download":
        download_page()
    elif selected2 == "Contact":
        contact_page()
    elif selected2 == 'Predict':
        predict_page()


if __name__ == "__main__":
    main()




















