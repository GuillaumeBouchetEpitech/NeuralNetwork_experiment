

TARGET_NAME=	exec
TARGET_DIR=		./bin
TARGET_PATHNAME= 	$(TARGET_DIR)/$(TARGET_NAME)

####

####


SRC_DIR=	./src
SRC=	\
	$(SRC_DIR)/main.cpp	\
	$(SRC_DIR)/machine-learning/Neuron.cpp \
	$(SRC_DIR)/machine-learning/NeuralNetwork.cpp \
	$(SRC_DIR)/utilities/RandomNumberGenerator.cpp \
	$(SRC_DIR)/utilities/TrainingData.cpp

OBJ_DIR=	./obj
OBJ=		$(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC))



#######


CXXFLAGS+=	-Wall -W -Wextra -Wunused
CXXFLAGS+=	-O3
CXXFLAGS+=	-std=c++20
CXXFLAGS+=	-I./

LDFLAGS=	-O3


#######


CXX=		g++
RM=			rm -rf


#######


all:			app

ensurefolders:
					@mkdir -p `dirname $(TARGET_PATHNAME)`

app:			ensurefolders $(OBJ)
					$(CXX) $(OBJ) -o $(TARGET_PATHNAME) $(LDFLAGS)

#

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo ' --> processing:' $<
	@$(CXX) $(CXXFLAGS) $< -c -o $@

#

clean:
					$(RM) $(OBJ)

fclean:		clean
					$(RM) $(TARGET_DIR)

re:				fclean all

.PHONY:		all clean fclean re
