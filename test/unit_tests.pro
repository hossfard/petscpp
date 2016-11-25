TEMPLATE = app
TARGET   = unittests

CONFIG -= qt
QMAKE_CXXFLAGS += -std=c++11

DEPENPATH += .
INCLUDEPATH += ../../eigen
INCLUDEPATH += $(GTEST_DIR)/include
LIBS += -L$(GTEST_DIR)/lib -lgtest_main -lpthread

INCLUDEPATH += $(PETSC_CC_INCLUDES)
LIBS += -lm $(PETSC_KSP_LIB)

INCLUDEPATH += ../

HEADERS +=

SOURCES += main.cpp \
           vector_unittest.cpp \
           matrix_unittest.cpp \
           utility_unittest.cpp \
           ../Petscpp/Vector.cpp \
           ../Petscpp/Matrix.cpp \
           ../Petscpp/IndexSet.cpp
