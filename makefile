OMPICC = g++-6
CC = mpicc
OPT = -Ofast -march=native -mtune=native
CFLAGS = -Wall $(OPT) -fopenmp
LDFLAGS = -Wall

SRCS=$(wildcard *.c)
OBJS=$(patsubst %.c,%.o,$(SRCS))
EXEC=$(patsubst %.c,%,$(SRCS))
report = $(patsubst %.c,%.optrpt,$(SRCS))

all: $(EXEC) $(OBJS)

%: %.o
	MPICH_CC=$(OMPICC) $(CC) $(CFLAGS) -o $@ $<

%.o: %.c
	MPICH_CC=$(OMPICC) $(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(EXEC) $(OBJS) $(report)
