OBJDIR=.obj
CFLAGS=`pkg-config --cflags jsoncpp libcurl` -std=c++11 -Wall
LIBS  =`pkg-config --libs jsoncpp libcurl` -lstdc++

all: dirs random_agent lib/gym_binding.a
.PHONY: clean all dirs

random_agent: .obj/random_agent.o .obj/gym_binding.o
	gcc -o $@ $^ $(LIBS)
	
lib/gym_binding.a: .obj/gym_binding.o
	ar rcs $@ $^

$(OBJDIR)/%.o: %.cpp
	g++ $(CFLAGS) -c $< -o $@

dirs:
	mkdir -p $(OBJDIR) lib

clean:
	rm -rf $(OBJDIR) lib/*

