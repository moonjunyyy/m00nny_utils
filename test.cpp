#include <iostream>
#include <memory>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::unique_ptr<int> p(new int(42));
    std::cout << *p << std::endl;
    int i = 36;
    auto pi = std::make_unique<int>(i);
    std::cout << *pi << std::endl;
    return 0;
}