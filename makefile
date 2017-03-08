MPICC = icc
CC = mpicc
OPT = -Ofast -fast -march=native -mtune=native -std=c99
CFLAGS = -Wall $(OPT) -fopenmp
LDFLAGS = -Wall

SRCS=$(wildcard *.c)
OBJS=$(patsubst %.c,%.o,$(SRCS))
EXEC=$(patsubst %.c,%,$(SRCS))
report = $(patsubst %.c,%.optrpt,$(SRCS))

all: $(EXEC) $(OBJS)

%: %.o
	MPICH_CC=$(MPICC) $(CC) $(CFLAGS) -o $@ $<

%.o: %.c
	MPICH_CC=$(MPICC) $(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(EXEC) $(OBJS) $(report)
