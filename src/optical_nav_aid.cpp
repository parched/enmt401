/* OPtical navigation aid by James Duley <jagduley@gmail.com> */

#include <stdlib.h>
#include <argp.h>

const char *argp_program_version = "Optical Navigation Aid v?";
const char *argp_program_bug_address = "<jagduley@gmail.com>";

/* Program documentation. */
static char doc[] = 
"An optical navigation aid for a UAV";

static struct argp argp = {0, 0, 0, doc};

int main(int argc, char **argv) {
	argp_parse (&argp, argc, argv, 0, 0, 0);

	return 0;
}
