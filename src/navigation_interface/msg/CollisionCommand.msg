# Private Tracking/Monitor protocol. The Monitor echoes the request ID verbatim.
uint64 request_id
geometry_msgs/Twist velocity
# Response only: this velocity was evaluated with a usable Monitor scan and TF.
bool input_valid
