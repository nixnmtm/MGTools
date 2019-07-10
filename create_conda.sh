#!/usr/bin/env bash

# author: Nixon
# reproducibly create conda env

read -p "Create new conda env (y/n)?" CONT

if [[ "${CONT}" == "n" ]]; then
  echo "exit";
else
# user chooses to create conda env
# prompt user for conda env name
  echo "Creating new conda environment, choose name"
  read input_variable
  echo "Name $input_variable was chosen";

  # Create environment.yml or not
   echo "installing base packages"
   conda config --add channels conda-forge
   conda create --name ${input_variable}\
   python=3.6 ipykernel mdanalysis nglview plotly
echo "to exit: source deactivate"
fi