# from .utils import Logger
# from .utils import yaml_loader, load_class
# from torch.optim import Adam


# class Trainer:
#     def __init__(self, conf, devices, logger):
#         pass
#         """
#         load dataset
#         load  model
#         load loss
#         load train objects --> optim,schedular
#         """

#     def _load_training_objects(self):
#         if "optimizer" in self.conf:
#             optim_class = load_class(self.conf.optimizer.target)
#             self.optimizer = optim_class(
#                 self.model.parameters(), **dict(self.conf.optimizer.params)
#             )
#         else:
#             self.optimizer = Adam(self.model.parameters())
#         self.logger.info(f"Using optimizer {self.optimizer.__class__.__name__}")

#         if "lr_scheduler" in self.conf:
#             scheduler_class = load_class(self.conf.lr_scheduler.target)
#             self.lr_scheduler = scheduler_class(
#                 self.optimizer, **dict(self.conf.lr_scheduler.params)
#             )
#             self.logger.info(f"Using scheduler {self.lr_scheduler.__class__.__name__}")
#         else:
#             self.lr_scheduler = None
#             self.logger.info("Not using any scheduler")

#     def _load_loss(self):
#         loss_class = load_class(self.conf.loss.target)
#         if "local_device_maps" in self.conf.loss.keys():
#             device = [
#                 self.devices[local_id] for local_id in self.conf.loss.local_device_maps
#             ][0]
#         else:
#             device = self.devices[0]
#         self.logger.info(f"Loss: using device: {device}")
#         if "params" in self.conf.loss.keys():
#             self.loss = loss_class(
#                 **dict(self.conf.loss.params),
#                 device=device,
#                 logger=self.logger,
#             )
#         else:
#             self.loss = loss_class(device=device, logger=self.logger)

#     def val_step(self):
#         pass

#     def train_step(self):
#         pass

#     def fit(self):
#         pass
#         """
#         train loop

#         """
