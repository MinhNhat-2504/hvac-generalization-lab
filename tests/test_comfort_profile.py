from datetime import timezone
import unittest

import pandas as pd
import pytz

from hvaclab.comfort_profile import split_comfort_loss
from smart_buildings.smart_control.proto import smart_control_reward_pb2
from smart_buildings.smart_control.reward.setpoint_energy_carbon_regret import SetpointEnergyCarbonRegretFunction
from smart_buildings.smart_control.simulator.setpoint_schedule import SetpointSchedule
from smart_buildings.smart_control.utils.conversion_utils import pandas_to_proto_timestamp


class ComfortProfileTest(unittest.TestCase):
    def test_naive_simulation_clock_must_not_be_converted_as_aware_utc(self):
        timestamp=pd.Timestamp("2023-04-24 06:00")
        schedule=SetpointSchedule(morning_start_hour=6,evening_start_hour=19,
                                  comfort_temp_window=(294.,297.),eco_temp_window=(289.,298.),
                                  time_zone="US/Pacific")
        self.assertTrue(schedule.is_comfort_mode(timestamp))
        converted=pandas_to_proto_timestamp(timestamp).ToDatetime(tzinfo=timezone.utc).astimezone(pytz.timezone("US/Pacific"))
        self.assertEqual(converted.hour,23)
        self.assertEqual(timestamp.hour,6)

    def test_direction_split_respects_occupancy_and_quadratic_excursions(self):
        info=smart_control_reward_pb2.RewardInfo()
        info.end_timestamp.seconds=300
        for key,temp,occupancy in (("cold",288.,2.),("hot",300.,1.),("inside",295.,3.)):
            z=info.zone_reward_infos[key]
            z.heating_setpoint_temperature=294.
            z.cooling_setpoint_temperature=297.
            z.zone_air_temperature=temp
            z.average_occupancy=occupancy
        reward=SetpointEnergyCarbonRegretFunction.__new__(SetpointEnergyCarbonRegretFunction)
        reward._productivity_deviation_ref=5.
        reward._max_productivity_personhour_usd=300.
        cold,hot=split_comfort_loss(info,.2,reward)
        self.assertAlmostEqual(cold+hot,.2)
        self.assertAlmostEqual(cold,.2*(2*1.04)/(2*1.04+.6))


if __name__ == "__main__":
    unittest.main()
