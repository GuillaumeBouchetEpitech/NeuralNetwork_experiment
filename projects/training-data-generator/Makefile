

TARGET_NAME=	exec
TARGET_DIR=		./bin
TARGET_PATHNAME= 	$(TARGET_DIR)/$(TARGET_NAME)

####


SRC_DIR=	./src
SRC+=			$(SRC_DIR)/main.cpp
SRC+=			$(SRC_DIR)/utilities/RandomNumberGenerator.cpp

OBJ_DIR=	./obj
OBJ=		$(patsubst %.cpp, $(OBJ_DIR)/%.o, $(SRC))


#######


CXXFLAGS+=	-Wall -W -Wextra -Wunused
CXXFLAGS+=	-O3
CXXFLAGS+=	-std=c++11
CXXFLAGS+=	-I./

LDFLAGS=	-O3


#######


CXX=		g++
RM=			rm -rf


#######


all:	ensurefolders	\
			app


ensurefolders:
					@mkdir -p `dirname $(TARGET_PATHNAME)`

app:		 ensurefolders $(OBJ)
					$(CXX) $(OBJ) -o $(TARGET_PATHNAME) $(LDFLAGS)

#

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	@echo ' --> processing:' $<
	@$(CXX) $(CXXFLAGS) $< -c -o $@

#

clean:
					$(RM) $(OBJ_DIR)

fclean:		clean
					$(RM) $(TARGET_PATHNAME)

re:				fclean all

.PHONY:		ensurefolders app all clean fclean re
