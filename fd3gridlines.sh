#!/bin/zsh
./fd3grid < "$1"inheta > "$1"outheta
./fd3grid < "$1"inhzeta > "$1"outhzeta
./fd3grid < "$1"inhdelta > "$1"outhdelta
./fd3grid < "$1"inhgamma > "$1"outhgamma
./fd3grid < "$1"inhbeta > "$1"outhbeta
./fd3grid < "$1"inheI+II4026 > "$1"outheI+II4026
./fd3grid < "$1"inheI4471 > "$1"outheI4471
./fd3grid < "$1"inheII4200 > "$1"outheII4200
./fd3grid < "$1"inheII4541 > "$1"outheII4541
./fd3grid < "$1"inheII4686 > "$1"outheII4686
