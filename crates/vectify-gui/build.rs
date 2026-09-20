//! Embeds the application icon into `vectify-gui.exe` on Windows, so Explorer,
//! shortcuts and the taskbar show it. (The window's own icon is set at runtime
//! from `assets/icon.png`; this is what makes the *file* show it.)
//!
//! Does nothing on other platforms.

use embed_resource::CompilationResult;

fn main() {
    // Cargo otherwise re-runs build scripts on any change in the package, and
    // emitting any of these turns that off, so list everything that matters --
    // including the icon, which `app.rc` references but Cargo cannot see.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=app.rc");
    println!("cargo:rerun-if-changed=../../assets/icon.ico");

    match embed_resource::compile("app.rc", embed_resource::NONE) {
        CompilationResult::NotWindows | CompilationResult::Ok => {}
        // No resource compiler (rc.exe or windres) on this machine. The icon is
        // cosmetic, so don't break the build over it, but do say so: without a
        // warning the exe would just quietly ship with the default icon.
        CompilationResult::NotAttempted(why) => {
            println!("cargo:warning=vectify-gui.exe will have no file icon: {why}");
        }
        failed @ CompilationResult::Failed(_) => panic!("{failed}"),
    }
}
