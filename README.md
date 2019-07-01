# Molecular Graph Theory

*This projects aims to depict the behaviour of molecules like proteins and nucleic acids*

In molecular graph theory, edges that connect the nodes of amino acid residues are calculated 
from atomic fluctuations. The adjacency matrix (A) and the degree matrix (D) of the weighted 
graph are also defined based on the calculated edges. The molecular graph was then characterized 
by analyzing the unoriented Laplacian matrix, $(K=A+D)$. We showed that this framework 
can be used to effectively dissect the inter-reside couplings due to backbone-backbone, 
backbone-sidechain, and sidechain-sidechain interactions, hence revealing the mechanical 
architecture of a protein structure. We also analyze the functional implications of the MGT 
in extracting allosteric communication and functionally important residues in proteins.


### Dependencies

"Run below commands to setup the environment or run the .yml" 

conda config --add channels conda-forge \
conda install mdanalysis -c conda-forge \
conda install ipykernel \
conda install -c plotly plotly \
conda install nglview -c conda-forge \
python -m ipykernel install --user --name py36MGT --display-name "Python (py36MGT)"