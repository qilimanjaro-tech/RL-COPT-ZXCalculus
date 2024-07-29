from rl_agent import AgentGNN
import gym_zx
import gymnasium as gym
import torch
from torch_geometric.data import Batch, Data
import pyzx as zx

def basic_optimise(c):
    c1 = zx.basic_optimization(c.copy().split_phase_gates()).to_basic_gates()
    c2 = zx.basic_optimization(c.copy().split_phase_gates(), do_swaps=True).to_basic_gates()
    if c2.twoqubitcount() < c1.twoqubitcount(): return c2 # As this optimisation algorithm is targetted at reducting H-gates, we use the circuit with the smaller 2-qubit gate count here, either using SWAP rules or not.
    return c1
def flow_opt(c):

    c =zx.optimize.basic_optimization(zx.Circuit.from_graph(c.to_graph().copy()).split_phase_gates())
    g = c.to_graph()        
    zx.teleport_reduce(g)
    zx.to_graph_like(g)
    zx.flow_2Q_simp(g)
    c2 = zx.extract_simple(g).to_basic_gates()
    return basic_optimise(c2)

if __name__ == "__main__":
    SrH_env = gym.make("zx-v0", qubits=12, depth=None, env_id=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rl_agent = AgentGNN(None,device=device).to(device)
    rl_agent.load_state_dict(
            torch.load("/home/jnogue/qilimanjaro/Copt-cquere/rl-zx/state_dict_5x70_cquere_twoqubits.pt", map_location=torch.device("cpu"))
        )  
    rl_agent.eval()
    twoqubit_gates = 60
    for _ in range(100):
        obs0, reset_info = SrH_env.reset()
        policy_items, value_items = reset_info["graph_obs"]
        value_graph = [Data(x=value_items[0], edge_index=value_items[1], edge_attr=value_items[2])]
        policy_graph = [Data(x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3])]
        next_obs_graph = (Batch.from_data_list(policy_graph), Batch.from_data_list(value_graph))
        done = False
        while not done:
            
            action, _, _, value, logits, action_ids = rl_agent.get_action_and_value(next_obs_graph, device=device)
            next_obs, reward, done, deprecated, info = SrH_env.step(action_ids.cpu().numpy())
            policy_items, value_items = info["graph_obs"]
            value_graph = [Data(x=value_items[0], edge_index=value_items[1], edge_attr=value_items[2])]
            policy_graph = [Data(x=policy_items[0], edge_index=policy_items[1], edge_attr=policy_items[2], y=policy_items[3])]
            next_obs_graph = (Batch.from_data_list(policy_graph), Batch.from_data_list(value_graph))
            final_circuit = flow_opt(SrH_env.final_circuit)
            data = final_circuit.to_dict()
            if data["twoqubits"] < twoqubit_gates:
                twoqubit_gates = data["twoqubits"]
                qasm_string = final_circuit.to_qasm()
                filename = "final_circuit.qasm"
                with open(filename, "w") as file:
                    file.write(qasm_string)

