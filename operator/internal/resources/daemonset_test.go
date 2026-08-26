/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package resources

import (
	"testing"

	corev1 "k8s.io/api/core/v1"

	lmcachev1alpha1 "github.com/LMCache/LMCache/api/v1alpha1"
)

// findEnv returns the env var named name from the engine container, or nil.
func findEnv(envs []corev1.EnvVar, name string) *corev1.EnvVar {
	for i := range envs {
		if envs[i].Name == name {
			return &envs[i]
		}
	}
	return nil
}

func TestBuildDaemonSet_WorkerNodeIPInjectedOnlyWithCoordinator(t *testing.T) {
	// Without a coordinator the server registers nowhere, so no node identity
	// env is injected and the env count contract of other tests holds.
	engine := minimalEngine()
	ds := BuildDaemonSet(engine)
	c := ds.Spec.Template.Spec.Containers[0]
	if e := findEnv(c.Env, "LMCACHE_WORKER_NODE_IP"); e != nil {
		t.Fatalf("unexpected LMCACHE_WORKER_NODE_IP without coordinator: %+v", e)
	}

	url := "http://coordinator:9300"
	engine.Spec.Coordinator = &lmcachev1alpha1.CoordinatorConnectionSpec{URL: &url}
	ds = BuildDaemonSet(engine)
	c = ds.Spec.Template.Spec.Containers[0]

	worker := findEnv(c.Env, "LMCACHE_WORKER_NODE_IP")
	if worker == nil || worker.ValueFrom == nil || worker.ValueFrom.FieldRef == nil {
		t.Fatalf("expected LMCACHE_WORKER_NODE_IP from the downward API, got %+v", worker)
	}
	if worker.ValueFrom.FieldRef.FieldPath != "status.hostIP" {
		t.Fatalf("expected status.hostIP, got %s", worker.ValueFrom.FieldRef.FieldPath)
	}
	advertise := findEnv(c.Env, "LMCACHE_COORDINATOR_ADVERTISE_IP")
	if advertise == nil || advertise.ValueFrom == nil || advertise.ValueFrom.FieldRef == nil {
		t.Fatalf("expected LMCACHE_COORDINATOR_ADVERTISE_IP from the downward API, got %+v", advertise)
	}
	// The direct MP address stays the pod IP; the worker IP is metadata only.
	if advertise.ValueFrom.FieldRef.FieldPath != "status.podIP" {
		t.Fatalf("expected status.podIP, got %s", advertise.ValueFrom.FieldRef.FieldPath)
	}

	// An explicit advertiseIP suppresses the pod-IP injection but never the
	// worker-node identity.
	explicit := "203.0.113.7"
	engine.Spec.Coordinator.AdvertiseIP = &explicit
	c = BuildDaemonSet(engine).Spec.Template.Spec.Containers[0]
	if findEnv(c.Env, "LMCACHE_COORDINATOR_ADVERTISE_IP") != nil {
		t.Fatal("explicit advertiseIP must suppress the pod-IP env injection")
	}
	if findEnv(c.Env, "LMCACHE_WORKER_NODE_IP") == nil {
		t.Fatal("explicit advertiseIP must not suppress LMCACHE_WORKER_NODE_IP")
	}
}

func TestStartupProbeFailureThreshold(t *testing.T) {
	cases := []struct {
		name     string
		l1SizeGB float64
		want     int32
	}{
		{"tiny is floored", 10, 30},
		{"fractional is floored", 0.5, 30},
		{"at floor boundary", 150, 30}, // 150/5 = 30, not greater than the floor
		{"just above floor", 155, 31},  // 155/5 = 31
		{"300GB", 300, 60},
		{"1200GB", 1200, 240},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := startupProbeFailureThreshold(tc.l1SizeGB); got != tc.want {
				t.Fatalf("startupProbeFailureThreshold(%v) = %d, want %d",
					tc.l1SizeGB, got, tc.want)
			}
		})
	}
}

func TestBuildDaemonSet_StartupProbeScalesWithL1(t *testing.T) {
	cases := []struct {
		name              string
		l1SizeGB          float64
		wantFailThreshold int32
	}{
		{"small L1 keeps the default window", 10, 30},
		{"large L1 gets a proportional window", 1200, 240},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			engine := minimalEngine()
			engine.Spec.L1.SizeGB = tc.l1SizeGB

			ds := BuildDaemonSet(engine)
			containers := ds.Spec.Template.Spec.Containers
			if len(containers) == 0 {
				t.Fatalf("expected at least one container in the DaemonSet")
			}
			probe := containers[0].StartupProbe
			if probe == nil {
				t.Fatalf("expected a startup probe on the engine container")
			}
			if probe.FailureThreshold != tc.wantFailThreshold {
				t.Fatalf("StartupProbe.FailureThreshold = %d, want %d",
					probe.FailureThreshold, tc.wantFailThreshold)
			}
			// Only the threshold scales; the period and initial delay are unchanged.
			if probe.PeriodSeconds != 5 {
				t.Fatalf("StartupProbe.PeriodSeconds = %d, want 5", probe.PeriodSeconds)
			}
			if probe.InitialDelaySeconds != 5 {
				t.Fatalf("StartupProbe.InitialDelaySeconds = %d, want 5",
					probe.InitialDelaySeconds)
			}
		})
	}
}
