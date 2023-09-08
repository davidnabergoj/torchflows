__test_constants = {
    'batch_shape': [(1,), (2,), (5,), (2, 4), (5, 2, 3, 2)],
    'event_shape': [(2,), (3,), (2, 4), (100,), (3, 7, 2)],
    'context_shape': [None, (2,), (3,), (2, 4), (5,)],
    'input_event_shape': [(2,), (3,), (2, 4), (100,), (3, 7, 2)],
    'output_event_shape': [(2,), (3,), (2, 4), (100,), (3, 7, 2)],
    'n_predicted_parameters': [1, 2, 10, 50, 100]
}
