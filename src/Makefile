CC = gcc
CCFLAGS = -g
LINK = gcc
LINKFLAGS = 

all:	polarRTE

%.o:	%.c
	$(CC) $(CCFLAGS) -c $< -o $@

polarRTE:	main.o visit_writer.o	
	$(LINK) $(LINKFLAGS) main.o visit_writer.o -o polarRTE

clean:
	rm *.o polarRTE