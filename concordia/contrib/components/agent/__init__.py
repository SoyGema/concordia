# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of components contributed by users."""

# Make all submodules explicitly available for import resolution
from concordia.contrib.components.agent import choice_of_component
from concordia.contrib.components.agent import situation_representation_via_narrative

# Explicit exports for pytype
__all__ = [
    'choice_of_component',
    'situation_representation_via_narrative',
]
