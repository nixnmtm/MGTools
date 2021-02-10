# Rigidity Graph

*This projects aims to depict the chemically spcific behaviour of molecules like proteins and nucleic acids*

The inter-residue rigidity graphs are the *K* matrices of the edge weights between residues *I* and *J*, 
which is the sum over the bsENM spring constants that link their CG sites. This construction renders a 
signless Laplacian matrix.  Since bsENM springs can be categorized according to the CG site types as backbone-backbone, 
backbone-side-chain, or side-chain-side-chain, *K^BB*,*K^BS* and *K^SS* are constructed accordingly. Averaging over the 
rigidity graphs of trajectory segments gives the mean graph *K*, and the following analysis for its 
eigenvalues and eigenvectors is conducted to identify the statistically prominent (strong and persistent) modes during 
protein dynamics. Then by comparing the prominent modes of apo and holo rigidity graphs, allosteric responses even 
without conformational changes can be elucidated. 


### Dependencies

"Run below commands to setup the environment or run the .yml" 

conda config --add channels conda-forge \
conda install mdanalysis -c conda-forge \
conda install ipykernel \
conda install -c plotly plotly \
conda install nglview -c conda-forge \
python -m ipykernel install --user --name py36MGT --display-name "Python (py36MGT)"