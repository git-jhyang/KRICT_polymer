# Dataset description
This project utilizes two distinct datasets for different stages of the machine learning model development: pre-training and fine-tuning.

## Pre-Training: [QM9 Dataset](http://quantum-machine.org/datasets/)
For the pre-training process, the QM9 dataset was employed. 
The QM9 is a subset of the GDB-17 database, a large chemical space of small organic molecules. 
It comprises approximately 134,000 stable organic molecules. Each molecule in the dataset is modeled using Density Functional Theory (DFT), specifically B3LYP/6-31G(2df,p) based DFT. 
The dataset includes detailed information on ground state geometry and quantum chemical properties. 
These properties include dipole moment, polarimetry, enthalpy, among others, contributing to a comprehensive understanding of each molecule's chemical characteristics.

- Example data format

| id   | smiles | smiles_rlx | inchi                      | inchi_rlx                  | na | rc_a     | rc_b     | rc_c     | mu    | alpha | homo   | lumo  | gap   | R^2    | zpve      | U0         | U          | H          | G          | Cv   | U0_pa      | U_pa       | H_pa              | G_pa       |
|------|--------|------------|----------------------------|----------------------------|----|----------|----------|----------|-------|-------|--------|-------|-------|--------|-----------|------------|------------|------------|------------|------|------------|------------|-------------------|------------|
| gdb-1 | C      | C          | InChI=1S/CH4/h1H4          | InChI=1S/CH4/h1H4          | 5  | 157.7118 | 157.70997| 157.70699| 0.0   | 13.21 | -0.3877| 0.1171| 0.5048| 35.3641| 0.044749  | -40.47893  | -40.476062 | -40.475117 | -40.498597| 6.469| -8.095786  | -8.0952124 | -8.095023399999999| -8.0997194 |
| gdb-2 | N      | N          | InChI=1S/H3N/h1H3          | InChI=1S/H3N/h1H3          | 4  | 293.60975| 293.54111| 191.39397| 1.6256| 9.46  | -0.257 | 0.0829| 0.3399| 26.1563| 0.034358  | -56.525887 | -56.523026 | -56.522082 | -56.544961| 6.316| -14.13147175| -14.1307565| -14.1305205       | -14.13624025|


## Fine-Tuning: [Fluorinated Polymer Dataset](https://f-polymer.chemdx.org/)
For the fine-tuning process, a specialized dataset focusing on fluorinated polymers was utilized. 
The full dataset is hosted at https://f-polymer.chemdx.org/. However, it is currently not available for public access. 
Researchers or interested parties who wish to access the dataset are encouraged to make a request. For potential use and access discussions, please send an inquiry to jang@krict.re.kr.
Upon approval, the dataset will be shared under specific terms and conditions that ensure its proper use and maintain necessary confidentiality measures.

- Example data format

| ID      | SMILES_A                      | SMILES_B                      | SMILES_C          | SMILES_D      | SMILES_E         | FR_A | FR_B | FR_C | FR_D | FR_E | TG    |
|---------|-------------------------------|-------------------------------|-------------------|---------------|------------------|------|------|------|------|------|-------|
| FA-00142| CC(=C)C(=O)OCC(F)(F)C(F)F     | CCCCCCCCCCCCCCCCCCOC(=O)C(C)=C| COC(=O)C(C)=C     | CC(=C)C(O)=O  | CC(=C)C(=O)OCCO  | 2    | 1    | 4    | 1    | 2    | ? |
