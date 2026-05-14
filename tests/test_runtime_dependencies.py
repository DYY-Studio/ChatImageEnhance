import subprocess
import sys
import unittest
from unittest.mock import call, patch

from sandbox.runtime_dependencies import RuntimeDependencyManager


class RuntimeDependencyManagerOpenCVTests(unittest.TestCase):
    def test_cv2_and_opencv_requests_prefer_contrib_package(self):
        self.assertEqual(
            RuntimeDependencyManager._normalize_package_name("cv2"),
            "opencv-contrib-python",
        )
        self.assertEqual(
            RuntimeDependencyManager._normalize_package_name("opencv-python>=4.8"),
            "opencv-contrib-python>=4.8",
        )
        self.assertEqual(
            RuntimeDependencyManager._guess_package_from_module_name("cv2"),
            "opencv-contrib-python",
        )

    def test_pip_install_command_supports_constraints_and_no_deps(self):
        self.assertEqual(
            RuntimeDependencyManager._build_pip_install_command("example-pkg", constraint_path="constraints.txt"),
            [sys.executable, "-m", "pip", "install", "--constraint", "constraints.txt", "example-pkg"],
        )
        self.assertEqual(
            RuntimeDependencyManager._build_pip_install_command("example-pkg", no_deps=True),
            [sys.executable, "-m", "pip", "install", "--no-deps", "example-pkg"],
        )

    def test_installed_contrib_satisfies_opencv_variants_without_version_churn(self):
        with patch.object(RuntimeDependencyManager, "_get_installed_version", return_value="4.12.0"), \
                patch("sandbox.runtime_dependencies.importlib.util.find_spec", return_value=object()):
            self.assertEqual(
                RuntimeDependencyManager._requirement_satisfied("opencv-python==4.1.0"),
                (True, "opencv-contrib-python"),
            )

    def test_dynamic_install_falls_back_to_no_deps_for_blocked_opencv_dependency(self):
        manager = RuntimeDependencyManager({})
        conflict = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Cannot install because opencv-python<0 conflicts with requirements",
        )
        success = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch.object(RuntimeDependencyManager, "_requirement_satisfied", return_value=(False, "example-pkg")), \
                patch.object(RuntimeDependencyManager, "_get_installed_version", return_value=None), \
                patch.object(manager, "_best_effort_release_for_install"), \
                patch.object(manager, "_run_pip_install", side_effect=[conflict, success]) as run_install:
            manager.install_packages(["example-pkg"])

        run_install.assert_has_calls([call("example-pkg"), call("example-pkg", no_deps=True)])

    def test_direct_opencv_python_install_is_rewritten(self):
        manager = RuntimeDependencyManager({})
        success = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        with patch.object(RuntimeDependencyManager, "_requirement_satisfied", return_value=(False, "opencv-contrib-python")), \
                patch.object(RuntimeDependencyManager, "_get_installed_version", return_value=None), \
                patch.object(manager, "_best_effort_release_for_install"), \
                patch.object(manager, "_run_pip_install", return_value=success) as run_install:
            manager.install_packages(["opencv-python==4.9.0.80"])

        run_install.assert_called_once_with("opencv-contrib-python==4.9.0.80")


if __name__ == "__main__":
    unittest.main()
