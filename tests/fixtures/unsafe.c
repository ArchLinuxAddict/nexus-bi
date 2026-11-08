#include <stdlib.h>
#include <string.h>

void copy_input(char *destination, const char *source) {
    strcpy(destination, source);
}

int run_command(const char *command) {
    return system(command);
}

int main(int argc, char **argv) {
    char buffer[16];
    copy_input(buffer, argv[0]);
    return run_command("true");
}
