#include <algorithm>
#include <iostream>
#include <typeinfo>
#include "carla/client/Client.h"
#include "CarlaDataAccessLayer.hpp"
#include "InMemoryMap.hpp"
#include "carla/client/Waypoint.h"
#include "ActorReadState.hpp"
#include "carla/client/Actor.h"


void test_closest_waypoint(carla::SharedPtr<carla::client::ActorList> vehicle_list, carla::SharedPtr<carla::client::Map> world_map);

int main(){
    auto client_conn = carla::client::Client("localhost", 2000);
    std::cout<<"Connected with client object : "<<client_conn.GetClientVersion()<<std::endl;
    auto world = client_conn.GetWorld();
    auto world_map = world.GetMap();
    auto actorList = world.GetActors();
    auto vehicle_list = actorList->Filter("vehicle.*");
 
    

    test_closest_waypoint(vehicle_list, world_map);

//     for(auto  it = _begin ; it != _end; it++ ) {
//         //std::cout << &it<<"\n";
//         //auto current_location = (*it)->GetId;
//         std::cout << (*it)->GetId() <<"\n";
//     }
    // auto dao = traffic_manager::ReadActorState(vehicle_list);
    
    //  dao.getLocation(vehicle_list);
    

    return 0;
}

void test_closest_waypoint(carla::SharedPtr<carla::client::ActorList> vehicle_list ,carla::SharedPtr<carla::client::Map> world_map){

    auto actor_obj = traffic_manager::ActorReadState(vehicle_list);
    
    auto actor_location = actor_obj.getLocation(vehicle_list);

    

    auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
    auto topology = dao.getTopology();
    auto closest_waypoint_obj = traffic_manager::InMemoryMap(topology);
    for(auto it = actor_location.begin(); it != actor_location.end(); it++)
    {
        auto closest_waypoint = closest_waypoint_obj.getWaypoint(*it);


        auto current_location = (closest_waypoint)->distance(*it);

        std::cout << current_location << "\n";
    }

    //std::cout << typeid(closest_waypoint).name() <<"\n";



}

// void test_get_topology(carla::SharedPtr<carla::client::Map> world_map) {

//     auto dao = traffic_manager::CarlaDataAccessLayer(world_map);
//     auto topology = dao.getTopology();

//     typedef std::vector<
//         std::pair<
//             carla::SharedPtr<carla::client::Waypoint>,
//             carla::SharedPtr<carla::client::Waypoint>
//         >
//     > toplist;
//     for(toplist::iterator it = topology.begin(); it != topology.end(); it++) {
//         auto wp1 = it->first;
//         auto wp2 = it->second;
//         std::cout << "Segment end road IDs : " << wp1->GetRoadId() << " -- " << wp2->GetRoadId() << std::endl;
//     }
// }