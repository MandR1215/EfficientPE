# Efficient Preference Elicitation in Iterative Combinatorial Auctions with Many Participants

## Preliminaries
This code primarily utilizes the GitHub project by Weissteiner et al. [2022].

For code requirements and execution details, please refer to this project.


### MLCA Experiments
In our experiments, we adhere to the MLCA Experiments section of the aforementioned project. The number of bidders is designated as ID ranges in `simulation_mlca.py`:

```
value_model = create_value_model(value_model=domain, 
                                 local_bidder_ids=range(15),
                                 regional_bidder_ids=range(15, 35),  
                                 national_bidder_ids=range(35, 50))
```

The number of items can be modified through the `.jar` file. <br>
For scenarios with 196 items, we provide `sats-0.8.1_item196.jar`. <br>
To simulate settings with 98 items, use `sats-0.8.1.jar` (download it <a href="https://github.com/spectrumauctions/sats">here</a>).

### References
[Weissteiner et al., 2022] Jakob Weissteiner, Jakob Heiss, Julien Siems, and Sven Seuken. Monotone-value neural networks: Exploiting preference monotonicity in combinatorial assignment. In Lud De Raedt, editor, Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI-22, pages 541–548. International Joint Conferences on Artificial Intelligence Organization, 7 2022. Main Track.