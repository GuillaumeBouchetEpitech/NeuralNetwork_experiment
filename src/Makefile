

NAME=		exec


####


SRC=		main.cpp
# SRC=		main_old.cpp

OBJ=		$(SRC:.cpp=.o)


#######


# CXXFLAGS=	-Wall -W -Wextra -Wunused -O3 -std=c++11 -I./
CXXFLAGS=	-Wall -W -Wextra -Wunused -g3 -std=c++11 -I./

LDFLAGS=	


#######


CXX=		g++
RM=			rm -f


#######


$(NAME):	$(OBJ)
			$(CXX) $(OBJ) -o $(NAME) $(LDFLAGS)

all:		$(NAME)

clean:
			$(RM) $(OBJ)

fclean:		clean
			$(RM) $(NAME)

re:			fclean all

.PHONY:		all clean fclean re
