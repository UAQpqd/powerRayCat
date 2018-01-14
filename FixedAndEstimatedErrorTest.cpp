//
// Created by dev on 1/13/2018.
//

#include <iostream>
#include "FixedAndEstimatedErrorTest.hpp"

void PowerRayCat::FixedAndEstimatedErrorTest::showTestMessage(const std::string &message) {
    std::cout << "[F&E Test]\t" << message << std::endl;
}

void PowerRayCat::FixedAndEstimatedErrorTest::run() {
    showTestMessage("Iniatiating test...");
}
