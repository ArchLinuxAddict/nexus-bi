unsigned sum_to(unsigned limit) {
    unsigned total = 0;
    for (unsigned index = 0; index < limit; index++) {
        total += index;
    }
    return total;
}

int main(void) {
    return (int)sum_to(10);
}
