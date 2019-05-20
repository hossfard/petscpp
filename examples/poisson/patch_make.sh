#!/bin/bash

sed -i -e 's/-I\$(PETSC_CC_INCLUDES)/\$(PETSC_CC_INCLUDES)/g' Makefile
echo "include \$(PETSC_DIR)/conf/variables" >> Makefile
